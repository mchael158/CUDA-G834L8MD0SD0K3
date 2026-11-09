//! Priority queue with aging and Adaptive stream-count heuristics.
//!
//! The scheduler never talks to CUDA directly. It only decides *which* queued job
//! should be considered next and *how many* backend slots Adaptive mode may use.
//! Actual issue/completion lives in [`crate::runtime::CoreWarp::poll`].

use std::time::Instant;

use corewarp_core::{JobId, JobStatus, Operation, Priority, SchedulingPolicy};
use corewarp_telemetry::JobMetrics;

/// One job sitting in (or temporarily deferred from) the ready queue.
#[derive(Debug)]
pub struct QueuedJob {
    pub id: JobId,
    pub priority: Priority,
    pub op: Operation,
    /// Only `Queued` entries are eligible for [`Scheduler::pop_next`].
    pub status: JobStatus,
    /// Wall-clock enqueue time; used to fill `JobMetrics::queue_wait_us`.
    pub enqueued_at: Instant,
    /// Aging boost added to [`Priority`] so Low work cannot starve forever.
    /// Cap is 8 (see [`Scheduler::age_waiting`]).
    pub age_boost: u8,
    pub metrics: JobMetrics,
}

impl QueuedJob {
    /// Effective sort key: base priority plus aging boost.
    pub fn effective_priority(&self) -> u32 {
        self.priority as u32 + u32::from(self.age_boost)
    }
}

/// Priority queue + Adaptive concurrency controller.
pub struct Scheduler {
    pub policy: SchedulingPolicy,
    pub max_in_flight: usize,
    pub issue_width: usize,
    pub stream_count: usize,
    /// Streams currently allowed under Adaptive (always `== stream_count` for Fixed).
    /// Range is `1..=stream_count`.
    pub active_streams: usize,
    queue: Vec<QueuedJob>,
    next_id: u64,
    /// Completions since the last Adaptive adjustment (threshold = 32).
    completed_since_adapt: u64,
}

impl Scheduler {
    /// Construct with the given policy and concurrency limits.
    ///
    /// Fixed and Adaptive both start with `active_streams = stream_count`.
    pub fn new(
        policy: SchedulingPolicy,
        max_in_flight: usize,
        issue_width: usize,
        stream_count: usize,
    ) -> Self {
        Self {
            policy,
            max_in_flight,
            issue_width,
            stream_count,
            active_streams: stream_count,
            queue: Vec::new(),
            next_id: 1,
            completed_since_adapt: 0,
        }
    }

    /// Number of jobs still waiting in the queue (any status kept there).
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Push a new job and return its freshly allocated [`JobId`].
    pub fn enqueue(&mut self, priority: Priority, op: Operation) -> JobId {
        let id = JobId(self.next_id);
        self.next_id += 1;
        self.queue.push(QueuedJob {
            id,
            priority,
            op,
            status: JobStatus::Queued,
            enqueued_at: Instant::now(),
            age_boost: 0,
            metrics: JobMetrics::default(),
        });
        id
    }

    /// Called every `poll`: bump `age_boost` for still-queued jobs (cap 8).
    ///
    /// This is a simple starvation-avoidance heuristic — not a fairness proof.
    pub fn age_waiting(&mut self) {
        for job in &mut self.queue {
            if job.status == JobStatus::Queued && job.age_boost < 8 {
                job.age_boost = job.age_boost.saturating_add(1);
            }
        }
    }

    /// Remove and return the highest effective-priority Queued job, if any.
    pub fn pop_next(&mut self) -> Option<QueuedJob> {
        let idx = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, j)| j.status == JobStatus::Queued)
            .max_by_key(|(_, j)| j.effective_priority())
            .map(|(i, _)| i)?;
        Some(self.queue.remove(idx))
    }

    /// Observe a completion and maybe adjust `active_streams` under Adaptive.
    ///
    /// `queue_depth_hint` is typically `pending + in_flight` at completion time.
    /// Adjustment is conservative: only every 32 completes, ±1 stream, clamped
    /// to `[1, stream_count]`. Uses *our* depth signal only — never fake GPU time.
    pub fn note_completed(&mut self, queue_depth_hint: usize) {
        self.completed_since_adapt += 1;
        if self.policy != SchedulingPolicy::Adaptive {
            return;
        }
        // Conservative adjustment every 32 completes, driven only by our own depth.
        if self.completed_since_adapt < 32 {
            return;
        }
        self.completed_since_adapt = 0;
        if queue_depth_hint > self.active_streams * 4 && self.active_streams < self.stream_count {
            self.active_streams += 1;
        } else if queue_depth_hint < self.active_streams && self.active_streams > 1 {
            self.active_streams -= 1;
        }
    }

    /// Admission control against the configured in-flight cap.
    pub fn can_admit(&self, in_flight: usize) -> bool {
        in_flight < self.max_in_flight
    }

    /// Re-insert a deferred job at the front so the next poll retries promptly.
    ///
    /// Used when buffer conflicts or Busy slots prevent issue in the current wave.
    pub(crate) fn push_front(&mut self, job: QueuedJob) {
        self.queue.insert(0, job);
    }
}
