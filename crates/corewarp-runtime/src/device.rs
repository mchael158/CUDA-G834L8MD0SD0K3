//! Public Device / Buffer / Submission API.
//!
//! This module is the stable surface that packs (`kernels`, `ai`, `game`) and
//! examples should call. Internally it wraps [`CoreWarp`] so advanced benches can
//! still reach the lower-level runtime via [`Device::runtime_mut`].

use corewarp_core::{
    BackendKind, BufferId, DeviceInfo, Error, JobHandle, JobStatus, Operation, Priority, Result,
    SchedulingPolicy, SubmissionStatus,
};
use corewarp_telemetry::JobMetrics;

use crate::runtime::CoreWarp;

/// Construction knobs mirrored onto [`corewarp_core::RuntimeConfig`].
///
/// Prefer `Device::host()` / `Device::cuda(0)` for defaults; use `*_with` when a
/// bench needs a specific stream count, issue width, or Adaptive scheduling.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Logical CUDA streams / host slots.
    pub streams: usize,
    /// Maximum concurrently in-flight backend jobs.
    pub max_in_flight: usize,
    /// Jobs attempted per `poll` wave.
    pub issue_width: usize,
    /// Host memory-pool budget in bytes.
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

/// Handle to a CoreWarp runtime bound to either the host or CUDA backend.
///
/// Methods that mutate device state take `&mut self` so overlapping submits from
/// multiple threads are intentionally unsupported in 0.1 (single-threaded owner).
pub struct Device {
    inner: CoreWarp,
    info: DeviceInfo,
}

/// Lightweight view of a buffer owned by this device.
///
/// Copying a [`Buffer`] does **not** duplicate device memory; it only copies the
/// ID and cached length. Freeing one alias invalidates the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    id: BufferId,
    len: usize,
}

impl Buffer {
    /// Opaque ID used when building [`Operation`] variants manually.
    pub fn id(&self) -> BufferId {
        self.id
    }

    /// Element count (`f32`) established at upload/allocate time.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when `len == 0` (should not occur for successfully created buffers).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Token returned by [`Device::submit`] identifying one queued/running job.
pub struct Submission {
    handle: JobHandle,
}

impl Submission {
    /// Lower-level job handle for metrics / status queries.
    pub fn handle(&self) -> JobHandle {
        self.handle
    }
}

impl Device {
    /// Open a host-CPU reference device with default config.
    pub fn host() -> Result<Self> {
        Self::host_with(DeviceConfig::default())
    }

    /// Open a host device with explicit [`DeviceConfig`].
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

    /// Open CUDA device `index` with default config (`require_cuda = true`).
    ///
    /// 0.1 only accepts `index == 0`. Missing Toolkit / feature fails hard rather
    /// than silently falling back to host.
    pub fn cuda(index: u32) -> Result<Self> {
        Self::cuda_with(index, DeviceConfig::default())
    }

    /// Open CUDA device `index` with explicit config.
    pub fn cuda_with(index: u32, cfg: DeviceConfig) -> Result<Self> {
        if index != 0 {
            return Err(Error::Invalid("only device 0 is supported in 0.1"));
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
            return Err(Error::Backend("CUDA unavailable".into()));
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

    /// Static description of the opened backend.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// `"host"` or `"cuda"` — useful for example banners and benches.
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    /// Upload host data into a new managed buffer (`host_dirty = true`).
    pub fn upload(&mut self, data: &[f32]) -> Result<Buffer> {
        let id = self.inner.upload(data)?;
        Ok(Buffer {
            id,
            len: data.len(),
        })
    }

    /// Allocate a zero-initialized buffer of `len` elements.
    pub fn allocate(&mut self, len: usize) -> Result<Buffer> {
        let id = self.inner.alloc_zeros(len)?;
        Ok(Buffer { id, len })
    }

    /// Enqueue `op` at [`Priority::Normal`].
    pub fn submit(&mut self, op: Operation) -> Result<Submission> {
        let handle = self.inner.submit(op)?;
        Ok(Submission { handle })
    }

    /// Enqueue `op` with an explicit priority (High beats Normal beats Low).
    pub fn submit_with_priority(&mut self, priority: Priority, op: Operation) -> Result<Submission> {
        let handle = self.inner.submit_with_priority(priority, op)?;
        Ok(Submission { handle })
    }

    /// Poll once and map the job's status to [`SubmissionStatus`].
    pub fn poll_submission(&mut self, submission: &Submission) -> Result<SubmissionStatus> {
        self.inner.poll()?;
        match self.inner.status(&submission.handle)? {
            JobStatus::Completed => Ok(SubmissionStatus::Completed),
            JobStatus::Failed => Ok(SubmissionStatus::Failed),
            _ => Ok(SubmissionStatus::Pending),
        }
    }

    /// Block until `submission` completes; propagates [`Error::Failed`] on error.
    pub fn wait(&mut self, submission: &Submission) -> Result<()> {
        self.inner.wait(&submission.handle)
    }

    /// Poll until the queue is empty and no backend slots are busy.
    pub fn drain(&mut self) -> Result<()> {
        self.inner.drain()
    }

    /// Bring the buffer coherent on the host and return a clone of its contents.
    ///
    /// Internally polls until `in_flight == 0`, then asks the backend to D2H if
    /// `device_dirty`.
    pub fn download(&mut self, buffer: &Buffer) -> Result<Vec<f32>> {
        self.inner.download(buffer.id)
    }

    /// Free host (and device-resident) storage for `buffer`.
    pub fn free(&mut self, buffer: Buffer) -> Result<()> {
        self.inner.free_buffer(buffer.id)
    }

    /// Stage timings for a specific submission (CUDA events when available).
    pub fn job_metrics(&self, submission: &Submission) -> Result<JobMetrics> {
        self.inner.job_metrics(&submission.handle)
    }

    /// Snapshot of aggregate runtime counters.
    pub fn metrics(&self) -> corewarp_telemetry::RuntimeMetrics {
        self.inner.metrics()
    }

    /// Drain outstanding work and shut the backend down cleanly.
    pub fn shutdown(self) -> Result<()> {
        self.inner.shutdown()
    }

    /// Escape hatch to the legacy/internal [`CoreWarp`] (migration and benches).
    pub fn runtime_mut(&mut self) -> &mut CoreWarp {
        &mut self.inner
    }
}
