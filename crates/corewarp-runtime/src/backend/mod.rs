//! Backend trait and completion records shared by host and CUDA implementations.
//!
//! A backend owns a fixed number of *slots* (logical streams). The runtime asks a
//! free slot to `try_issue` a job and later drains finished work via
//! `poll_completions`. Download coherence and device-buffer recycling are also
//! backend responsibilities so the host path can be a pure no-op.

mod host;

#[cfg(feature = "cuda")]
mod cuda;

use corewarp_core::{BufferId, JobId, Operation, Result};
use corewarp_memory::MemoryManager;
use corewarp_telemetry::{JobMetrics, RuntimeMetrics};

pub(crate) use host::HostBackend;

#[cfg(feature = "cuda")]
pub(crate) use cuda::CudaBackend;

/// One finished job returned by [`Backend::poll_completions`].
pub(crate) struct CompletedWork {
    pub job_id: JobId,
    /// Backend-measured stage timings (zeros on host).
    pub metrics: JobMetrics,
    /// `Some(message)` marks the job as Failed; `None` means success.
    pub failed: Option<String>,
}

/// Pluggable execution engine behind [`crate::runtime::CoreWarp`].
pub(crate) trait Backend {
    /// Number of issue slots (streams).
    fn slot_count(&self) -> usize;
    /// How many slots currently hold in-flight work.
    fn in_flight(&self) -> usize;
    /// Attempt to launch `op` on `slot`. Return [`Error::Busy`] if the slot is taken.
    fn try_issue(
        &mut self,
        slot: usize,
        job_id: JobId,
        op: &Operation,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()>;
    /// Non-blocking harvest of finished slots.
    fn poll_completions(
        &mut self,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<Vec<CompletedWork>>;
    /// Ensure buffer `id` is coherent on the host before [`MemoryManager::download`].
    fn prepare_download(
        &mut self,
        id: BufferId,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()>;
    /// Drop any device-resident storage for `id` (host: no-op).
    fn free_device(&mut self, id: BufferId, metrics: &mut RuntimeMetrics) -> Result<()>;
    /// Release backend resources after the runtime has drained.
    fn shutdown(&mut self) -> Result<()>;
}
