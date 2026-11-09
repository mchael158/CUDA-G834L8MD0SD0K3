mod host;

#[cfg(feature = "cuda")]
mod cuda;

use corewarp_core::{BufferId, JobId, Operation, Result};
use corewarp_memory::MemoryManager;
use corewarp_telemetry::{JobMetrics, RuntimeMetrics};

pub(crate) use host::HostBackend;

#[cfg(feature = "cuda")]
pub(crate) use cuda::CudaBackend;

pub(crate) struct CompletedWork {
    pub job_id: JobId,
    pub metrics: JobMetrics,
    pub failed: Option<String>,
}

pub(crate) trait Backend {
    fn slot_count(&self) -> usize;
    fn in_flight(&self) -> usize;
    fn try_issue(
        &mut self,
        slot: usize,
        job_id: JobId,
        op: &Operation,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()>;
    fn poll_completions(
        &mut self,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<Vec<CompletedWork>>;
    fn prepare_download(
        &mut self,
        id: BufferId,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()>;
    fn free_device(&mut self, id: BufferId, metrics: &mut RuntimeMetrics) -> Result<()>;
    fn shutdown(&mut self) -> Result<()>;
}
