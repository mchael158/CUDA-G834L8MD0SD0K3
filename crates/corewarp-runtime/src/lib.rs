//! # corewarp-runtime
//!
//! Execution engine for CoreWarp: public Device API, priority scheduler, host and
//! (optional) CUDA backends, and the internal [`CoreWarp`] orchestrator.
//!
//! ## Module map
//! - [`device`] — `Device` / `Buffer` / `Submission` façade used by applications.
//! - [`runtime`] — queue + memory + conflict locks + backend dispatch loop.
//! - [`scheduler`] — priority queue with aging and Adaptive stream heuristics.
//! - [`backend`] — `HostBackend` always; `CudaBackend` behind feature `cuda`.
//!
//! Contract tests under `tests_contract` exercise the host backend only so CI can
//! run without a CUDA Toolkit installed.

mod backend;
mod device;
mod runtime;
mod scheduler;

pub use corewarp_core::{
    BackendKind, BufferId, DeviceInfo, Error, JobHandle, JobId, JobStatus, Operation, Priority,
    Result, RuntimeConfig, SchedulingPolicy, SubmissionStatus,
};
pub use corewarp_telemetry::{JobMetrics, RuntimeMetrics};
pub use device::{Buffer, Device, DeviceConfig, Submission};
pub use runtime::{CoreWarp, CoreWarpBuilder};

#[cfg(test)]
mod tests_contract;
