//! Runtime CoreWarp: Device, scheduling e backends.

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
