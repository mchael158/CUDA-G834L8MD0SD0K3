//! # CoreWarp — GPU Compute Toolkit (facade crate)
//!
//! This crate is the public entry point for applications. It re-exports the core
//! types, Device API, shared kernels, telemetry, and (optionally) the AI / Game
//! domain packs behind Cargo features.
//!
//! ## Quick start (host backend — always available)
//! ```ignore
//! use corewarp::prelude::*;
//! let mut device = Device::host()?;
//! let buf = device.upload(&[1.0, 2.0, 3.0])?;
//! let out = device.allocate(3)?;
//! let sub = device.submit(Operation::VectorScale {
//!     alpha: 2.0,
//!     input: buf.id(),
//!     output: out.id(),
//! })?;
//! device.wait(&sub)?;
//! let result = device.download(&out)?;
//! ```
//!
//! ## Features
//! - `cuda` — enable the CUDA driver backend (requires Toolkit + compatible driver).
//! - `ai` — re-export [`corewarp_ai`] as [`ai`].
//! - `game` — re-export [`corewarp_game`] as [`game`].
//!
//! Host-only builds remain useful for contract tests and CPU reference paths; they
//! never fabricate GPU timing metrics.

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

/// Minimal prelude for application code (`use corewarp::prelude::*`).
pub mod prelude {
    pub use crate::{
        Buffer, Device, DeviceConfig, Error, Operation, Priority, Result, Submission,
        SubmissionStatus,
    };
}
