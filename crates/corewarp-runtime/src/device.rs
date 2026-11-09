//! API pública Device / Buffer / Submission.

use corewarp_core::{
    BackendKind, BufferId, DeviceInfo, Error, JobHandle, JobStatus, Operation, Priority, Result,
    SchedulingPolicy, SubmissionStatus,
};
use corewarp_telemetry::JobMetrics;

use crate::runtime::CoreWarp;

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub streams: usize,
    pub max_in_flight: usize,
    pub issue_width: usize,
    pub memory_pool_bytes: usize,
    pub scheduling: SchedulingPolicy,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            streams: 4,
            max_in_flight: 32,
            issue_width: 2,
            memory_pool_bytes: 128 * 1024 * 1024,
            scheduling: SchedulingPolicy::Fixed,
        }
    }
}

pub struct Device {
    inner: CoreWarp,
    info: DeviceInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    id: BufferId,
    len: usize,
}

impl Buffer {
    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct Submission {
    handle: JobHandle,
}

impl Submission {
    pub fn handle(&self) -> JobHandle {
        self.handle
    }
}

impl Device {
    pub fn host() -> Result<Self> {
        Self::host_with(DeviceConfig::default())
    }

    pub fn host_with(cfg: DeviceConfig) -> Result<Self> {
        let inner = CoreWarp::builder()
            .backend(BackendKind::Host)
            .streams(cfg.streams)
            .max_in_flight(cfg.max_in_flight)
            .issue_width(cfg.issue_width)
            .memory_pool_bytes(cfg.memory_pool_bytes)
            .scheduling(cfg.scheduling)
            .build()?;
        Ok(Self {
            info: DeviceInfo {
                backend: "host",
                index: 0,
                name: "HostCPU".into(),
            },
            inner,
        })
    }

    pub fn cuda(index: u32) -> Result<Self> {
        Self::cuda_with(index, DeviceConfig::default())
    }

    pub fn cuda_with(index: u32, cfg: DeviceConfig) -> Result<Self> {
        if index != 0 {
            return Err(Error::Invalid("apenas device 0 no 0.1"));
        }
        let inner = CoreWarp::builder()
            .backend(BackendKind::Cuda)
            .require_cuda(true)
            .streams(cfg.streams)
            .max_in_flight(cfg.max_in_flight)
            .issue_width(cfg.issue_width)
            .memory_pool_bytes(cfg.memory_pool_bytes)
            .scheduling(cfg.scheduling)
            .build()?;
        if inner.backend_name() != "cuda" {
            return Err(Error::Backend("CUDA indisponível".into()));
        }
        Ok(Self {
            info: DeviceInfo {
                backend: "cuda",
                index,
                name: format!("CUDA:{index}"),
            },
            inner,
        })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    pub fn upload(&mut self, data: &[f32]) -> Result<Buffer> {
        let id = self.inner.upload(data)?;
        Ok(Buffer {
            id,
            len: data.len(),
        })
    }

    pub fn allocate(&mut self, len: usize) -> Result<Buffer> {
        let id = self.inner.alloc_zeros(len)?;
        Ok(Buffer { id, len })
    }

    pub fn submit(&mut self, op: Operation) -> Result<Submission> {
        let handle = self.inner.submit(op)?;
        Ok(Submission { handle })
    }

    pub fn submit_with_priority(&mut self, priority: Priority, op: Operation) -> Result<Submission> {
        let handle = self.inner.submit_with_priority(priority, op)?;
        Ok(Submission { handle })
    }

    pub fn poll_submission(&mut self, submission: &Submission) -> Result<SubmissionStatus> {
        self.inner.poll()?;
        match self.inner.status(&submission.handle)? {
            JobStatus::Completed => Ok(SubmissionStatus::Completed),
            JobStatus::Failed => Ok(SubmissionStatus::Failed),
            _ => Ok(SubmissionStatus::Pending),
        }
    }

    pub fn wait(&mut self, submission: &Submission) -> Result<()> {
        self.inner.wait(&submission.handle)
    }

    pub fn drain(&mut self) -> Result<()> {
        self.inner.drain()
    }

    pub fn download(&mut self, buffer: &Buffer) -> Result<Vec<f32>> {
        self.inner.download(buffer.id)
    }

    pub fn free(&mut self, buffer: Buffer) -> Result<()> {
        self.inner.free_buffer(buffer.id)
    }

    pub fn job_metrics(&self, submission: &Submission) -> Result<JobMetrics> {
        self.inner.job_metrics(&submission.handle)
    }

    pub fn metrics(&self) -> corewarp_telemetry::RuntimeMetrics {
        self.inner.metrics()
    }

    pub fn shutdown(self) -> Result<()> {
        self.inner.shutdown()
    }

    /// Acesso ao runtime legado (migração / benches).
    pub fn runtime_mut(&mut self) -> &mut CoreWarp {
        &mut self.inner
    }
}
