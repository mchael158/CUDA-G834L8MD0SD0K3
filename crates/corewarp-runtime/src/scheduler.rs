use std::time::Instant;

use corewarp_core::{JobId, JobStatus, Operation, Priority, SchedulingPolicy};
use corewarp_telemetry::JobMetrics;

#[derive(Debug)]
pub struct QueuedJob {
    pub id: JobId,
    pub priority: Priority,
    pub op: Operation,
    pub status: JobStatus,
    pub enqueued_at: Instant,
    pub age_boost: u8,
    pub metrics: JobMetrics,
}

impl QueuedJob {
    pub fn effective_priority(&self) -> u32 {
        self.priority as u32 + u32::from(self.age_boost)
    }
}

pub struct Scheduler {
    pub policy: SchedulingPolicy,
    pub max_in_flight: usize,
    pub issue_width: usize,
    pub stream_count: usize,
    /// Streams ativas sob Adaptive (1..=stream_count).
    pub active_streams: usize,
    queue: Vec<QueuedJob>,
    next_id: u64,
    completed_since_adapt: u64,
}

impl Scheduler {
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

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

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

    pub fn age_waiting(&mut self) {
        for job in &mut self.queue {
            if job.status == JobStatus::Queued && job.age_boost < 8 {
                job.age_boost = job.age_boost.saturating_add(1);
            }
        }
    }

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

    pub fn note_completed(&mut self, queue_depth_hint: usize) {
        self.completed_since_adapt += 1;
        if self.policy != SchedulingPolicy::Adaptive {
            return;
        }
        // Ajuste conservador a cada 32 completes, só com medições próprias.
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

    pub fn can_admit(&self, in_flight: usize) -> bool {
        in_flight < self.max_in_flight
    }

    pub(crate) fn push_front(&mut self, job: QueuedJob) {
        self.queue.insert(0, job);
    }
}
