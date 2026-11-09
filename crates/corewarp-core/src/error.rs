//! Error types shared by every CoreWarp crate.
//!
//! Errors are intentionally coarse-grained: they classify *why* a call failed so
//! the Device API can map them to user-facing outcomes (invalid input, OOM,
//! backpressure, incomplete work, backend failures) without leaking CUDA or host
//! implementation details into higher layers.

use std::fmt;

/// Uniform failure modes for CoreWarp APIs.
///
/// Prefer returning these over panicking. Backend-specific diagnostics should be
/// stringified into [`Error::Backend`] or [`Error::Failed`] rather than inventing
/// new variants per driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Caller-provided arguments violate a contract (empty buffers, mismatched
    /// lengths, unsupported device index, etc.). The `&'static str` is a short
    /// English reason suitable for logs and asserts.
    Invalid(&'static str),
    /// Host or device memory budget exhausted (pool charge would exceed limit).
    Oom,
    /// Lower-level backend/driver failure (CUDA init, NVRTC compile, memcpy, …).
    Backend(String),
    /// Resources are temporarily unavailable (slot busy, buffer still in-flight).
    /// Callers should poll/retry rather than treat this as a hard failure.
    Busy,
    /// Requested data is not yet coherent on the host (job still running, or
    /// device-dirty without a completed download sync).
    NotReady,
    /// A submitted job finished in a failed state; message carries the cause.
    Failed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) => write!(f, "invalid: {msg}"),
            Self::Oom => write!(f, "out of memory"),
            Self::Backend(msg) => write!(f, "backend: {msg}"),
            Self::Busy => write!(f, "backpressure: resources busy"),
            Self::NotReady => write!(f, "not ready"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, Error>;
