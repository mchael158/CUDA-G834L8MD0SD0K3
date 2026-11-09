//! # corewarp-core
//!
//! Shared, CUDA-free foundation types for the CoreWarp GPU Compute Toolkit.
//!
//! This crate intentionally has **zero** GPU / driver dependencies so that host-only
//! builds, tests, and documentation can depend on the same type system used by the
//! CUDA backend. Everything that describes *what* a job is (operations, priorities,
//! buffer identities, runtime configuration) lives here; *how* those jobs execute
//! lives in `corewarp-runtime`.
//!
//! ## What you get
//! - [`Error`] / [`Result`]: uniform error surface across all crates.
//! - Identifiers: [`JobId`], [`BufferId`], [`JobHandle`].
//! - Scheduling enums: [`Priority`], [`JobStatus`], [`SubmissionStatus`], [`SchedulingPolicy`].
//! - Backend selection: [`BackendKind`], [`RuntimeConfig`], [`DeviceInfo`].
//! - Typed workloads: [`Operation`] with shape validation and conflict detection.
//!
//! ## Design notes
//! Buffer and job IDs are opaque `u64` wrappers. Callers must not assume dense
//! packing or reuse of IDs after free; the memory manager owns lifetime semantics.

mod error;
mod types;

pub use error::{Error, Result};
pub use types::*;
