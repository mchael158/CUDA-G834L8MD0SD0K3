use std::collections::HashMap;

use corewarp_core::{
    BackendKind, BufferId, Error, JobHandle, JobStatus, Operation, Priority, Result, RuntimeConfig,
    SchedulingPolicy,
};
use corewarp_memory::MemoryManager;
use corewarp_telemetry::{JobMetrics, RuntimeMetrics};

use crate::backend::{Backend, HostBackend};
use crate::scheduler::Scheduler;

#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;

enum BackendBox {
    Host(HostBackend),
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
}

impl BackendBox {
    fn as_mut(&mut self) -> &mut dyn Backend {
        match self {
            Self::Host(b) => b,
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Host(_) => "host",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
        }
    }
}

struct JobRecord {
    status: JobStatus,
    fail: Option<String>,
    metrics: JobMetrics,
}

/// Runtime interno: fila, memória e backends.
pub struct CoreWarp {
    memory: MemoryManager,
    scheduler: Scheduler,
    backend: BackendBox,
    jobs: HashMap<u64, JobRecord>,
    metrics: RuntimeMetrics,
    active_reads: HashMap<BufferId, u32>,
    active_writes: HashMap<BufferId, u32>,
    inflight_buffers: HashMap<u64, (Vec<BufferId>, Vec<BufferId>)>,
}

pub struct CoreWarpBuilder {
    cfg: RuntimeConfig,
    require_cuda: bool,
}

impl CoreWarp {
    pub fn builder() -> CoreWarpBuilder {
        CoreWarpBuilder {
            cfg: RuntimeConfig::default(),
            require_cuda: false,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn metrics(&self) -> RuntimeMetrics {
        self.metrics
    }

    pub fn job_metrics(&self, handle: &JobHandle) -> Result<JobMetrics> {
        self.jobs
            .get(&handle.id.0)
            .map(|j| j.metrics)
            .ok_or(Error::Invalid("job inexistente"))
    }

    pub fn upload(&mut self, data: &[f32]) -> Result<BufferId> {
        self.memory.upload(data, &mut self.metrics)
    }

    pub fn alloc_zeros(&mut self, len: usize) -> Result<BufferId> {
        self.memory.alloc_zeros(len, &mut self.metrics)
    }

    pub fn submit(&mut self, op: Operation) -> Result<JobHandle> {
        self.submit_with_priority(Priority::Normal, op)
    }

    pub fn submit_with_priority(&mut self, priority: Priority, op: Operation) -> Result<JobHandle> {
        op.validate_shapes(|id| self.memory.len_of(id))?;
        let ids = op.buffer_ids();
        self.memory.retain(&ids)?;
        let id = self.scheduler.enqueue(priority, op);
        self.jobs.insert(
            id.0,
            JobRecord {
                status: JobStatus::Queued,
                fail: None,
                metrics: JobMetrics::default(),
            },
        );
        self.metrics.jobs_submitted += 1;
        Ok(JobHandle { id })
    }

    pub fn status(&self, handle: &JobHandle) -> Result<JobStatus> {
        self.jobs
            .get(&handle.id.0)
            .map(|j| j.status)
            .ok_or(Error::Invalid("job inexistente"))
    }

    pub fn is_complete(&self, handle: &JobHandle) -> Result<bool> {
        Ok(matches!(
            self.status(handle)?,
            JobStatus::Completed | JobStatus::Failed
        ))
    }

    pub fn poll(&mut self) -> Result<usize> {
        self.metrics.polls += 1;
        self.scheduler.age_waiting();

        let completed = self
            .backend
            .as_mut()
            .poll_completions(&mut self.memory, &mut self.metrics)?;
        let n_done = completed.len();
        for work in completed {
            if let Some((reads, writes)) = self.inflight_buffers.remove(&work.job_id.0) {
                for id in reads {
                    Self::release_lock(&mut self.active_reads, id);
                }
                for id in writes {
                    Self::release_lock(&mut self.active_writes, id);
                }
            }
            if let Some(rec) = self.jobs.get_mut(&work.job_id.0) {
                let queue_wait_us = rec.metrics.queue_wait_us;
                rec.metrics = work.metrics;
                rec.metrics.queue_wait_us = queue_wait_us;
                if let Some(msg) = work.failed {
                    rec.status = JobStatus::Failed;
                    rec.fail = Some(msg);
                    self.metrics.jobs_failed += 1;
                } else {
                    rec.status = JobStatus::Completed;
                    self.metrics.jobs_completed += 1;
                }
            }
            let depth = self.scheduler.pending() + self.backend.as_mut().in_flight();
            self.scheduler.note_completed(depth);
        }

        let width = self.scheduler.issue_width;
        let active = self
            .scheduler
            .active_streams
            .min(self.backend.as_mut().slot_count());

        let mut deferred = Vec::new();
        for _ in 0..width {
            if !self.scheduler.can_admit(self.backend.as_mut().in_flight()) {
                break;
            }
            let Some(mut job) = self.scheduler.pop_next() else {
                break;
            };
            job.metrics.queue_wait_us = job.enqueued_at.elapsed().as_micros() as u64;

            let active_reads: Vec<_> = self.active_reads.keys().copied().collect();
            let active_writes: Vec<_> = self.active_writes.keys().copied().collect();
            if job.op.conflicts_with(&active_reads, &active_writes) {
                deferred.push(job);
                continue;
            }

            let mut issued = false;
            for slot in 0..active {
                match self.backend.as_mut().try_issue(
                    slot,
                    job.id,
                    &job.op,
                    &mut self.memory,
                    &mut self.metrics,
                ) {
                    Ok(()) => {
                        let reads = job.op.reads();
                        let writes = job.op.writes();
                        for id in &reads {
                            *self.active_reads.entry(*id).or_insert(0) += 1;
                        }
                        for id in &writes {
                            *self.active_writes.entry(*id).or_insert(0) += 1;
                        }
                        self.inflight_buffers.insert(job.id.0, (reads, writes));
                        if let Some(rec) = self.jobs.get_mut(&job.id.0) {
                            rec.status = JobStatus::Running;
                            rec.metrics.queue_wait_us = job.metrics.queue_wait_us;
                        }
                        issued = true;
                        break;
                    }
                    Err(Error::Busy) => continue,
                    Err(e) => {
                        self.memory.release(&job.op.buffer_ids());
                        if let Some(rec) = self.jobs.get_mut(&job.id.0) {
                            rec.status = JobStatus::Failed;
                            rec.fail = Some(e.to_string());
                        }
                        self.metrics.jobs_failed += 1;
                        issued = true;
                        break;
                    }
                }
            }
            if !issued {
                deferred.push(job);
                break;
            }
        }
        for job in deferred.into_iter().rev() {
            self.scheduler.push_front(job);
        }
        Ok(n_done)
    }

    pub fn wait(&mut self, handle: &JobHandle) -> Result<()> {
        loop {
            if self.is_complete(handle)? {
                if self.status(handle)? == JobStatus::Failed {
                    let msg = self
                        .jobs
                        .get(&handle.id.0)
                        .and_then(|j| j.fail.clone())
                        .unwrap_or_else(|| "job failed".into());
                    return Err(Error::Failed(msg));
                }
                return Ok(());
            }
            self.poll()?;
        }
    }

    pub fn drain(&mut self) -> Result<()> {
        loop {
            self.poll()?;
            let pending = self.scheduler.pending();
            let inflight = self.backend.as_mut().in_flight();
            if pending == 0 && inflight == 0 {
                return Ok(());
            }
        }
    }

    pub fn download(&mut self, id: BufferId) -> Result<Vec<f32>> {
        loop {
            self.poll()?;
            let entry = self.memory.get(id)?;
            if entry.in_flight > 0 {
                continue;
            }
            self.backend
                .as_mut()
                .prepare_download(id, &mut self.memory, &mut self.metrics)?;
            return self.memory.download(id);
        }
    }

    pub fn free_buffer(&mut self, id: BufferId) -> Result<()> {
        self.backend
            .as_mut()
            .free_device(id, &mut self.metrics)?;
        self.memory.free(id)
    }

    pub fn shutdown(mut self) -> Result<()> {
        self.drain()?;
        self.backend.as_mut().shutdown()
    }

    fn release_lock(map: &mut HashMap<BufferId, u32>, id: BufferId) {
        if let Some(count) = map.get_mut(&id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                map.remove(&id);
            }
        }
    }
}

impl CoreWarpBuilder {
    pub fn backend(mut self, backend: BackendKind) -> Self {
        self.cfg.backend = backend;
        self
    }

    pub fn streams(mut self, n: usize) -> Self {
        self.cfg.stream_count = n;
        self
    }

    pub fn max_in_flight(mut self, n: usize) -> Self {
        self.cfg.max_in_flight = n;
        self
    }

    pub fn issue_width(mut self, n: usize) -> Self {
        self.cfg.issue_width = n;
        self
    }

    pub fn memory_pool_bytes(mut self, n: usize) -> Self {
        self.cfg.memory_pool_bytes = n;
        self
    }

    pub fn scheduling(mut self, p: SchedulingPolicy) -> Self {
        self.cfg.scheduling = p;
        self
    }

    pub fn require_cuda(mut self, yes: bool) -> Self {
        self.require_cuda = yes;
        self
    }

    pub fn build(self) -> Result<CoreWarp> {
        let cfg = self.cfg;
        let require_cuda = self.require_cuda;
        if cfg.stream_count == 0 || cfg.max_in_flight == 0 || cfg.issue_width == 0 {
            return Err(Error::Invalid("stream_count/max_in_flight/issue_width > 0"));
        }

        let backend = match cfg.backend {
            BackendKind::Host => BackendBox::Host(HostBackend::new(cfg.stream_count)),
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match CudaBackend::new(cfg.stream_count) {
                        Ok(b) => BackendBox::Cuda(b),
                        Err(e) if require_cuda => return Err(e),
                        Err(_) => BackendBox::Host(HostBackend::new(cfg.stream_count)),
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    if require_cuda {
                        return Err(Error::Backend("feature `cuda` não compilada".into()));
                    }
                    BackendBox::Host(HostBackend::new(cfg.stream_count))
                }
            }
        };

        Ok(CoreWarp {
            memory: MemoryManager::new(cfg.memory_pool_bytes),
            scheduler: Scheduler::new(
                cfg.scheduling,
                cfg.max_in_flight,
                cfg.issue_width,
                cfg.stream_count,
            ),
            backend,
            jobs: HashMap::new(),
            metrics: RuntimeMetrics::default(),
            active_reads: HashMap::new(),
            active_writes: HashMap::new(),
            inflight_buffers: HashMap::new(),
        })
    }
}
