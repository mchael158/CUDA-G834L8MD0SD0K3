//! CoreWarp — GPU Compute Toolkit.
//!
//! ```ignore
//! use corewarp::prelude::*;
//! let mut device = Device::host()?;
//! let buf = device.upload(&[1.0, 2.0, 3.0])?;
//! ```

pub use corewarp_core::{
    BackendKind, BufferId, DeviceInfo, Error, JobHandle, JobId, JobStatus, Operation, Priority,
    Result, RuntimeConfig, SchedulingPolicy, SubmissionStatus,
};
pub use corewarp_kernels::{
    histogram, prefix_sum, reduction_sum, saxpy, vector_add, vector_scale,
};
pub use corewarp_runtime::{
    Buffer, CoreWarp, CoreWarpBuilder, Device, DeviceConfig, JobMetrics, RuntimeMetrics, Submission,
};
pub use corewarp_telemetry;

#[cfg(feature = "ai")]
pub use corewarp_ai as ai;

#[cfg(feature = "game")]
pub use corewarp_game as game;

pub mod prelude {
    pub use crate::{
        Buffer, Device, DeviceConfig, Error, Operation, Priority, Result, Submission,
        SubmissionStatus,
    };
}
