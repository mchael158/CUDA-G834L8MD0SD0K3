//! # corewarp-telemetry
//!
//! Lightweight metrics structs shared by the runtime, memory pool, and backends.
//!
//! These are plain Copy aggregates updated in place during `poll` / issue / free.
//! They intentionally do **not** invent GPU timings on the host backend: host
//! `JobMetrics` stay at zero unless a real CUDA event measurement is available.

/// Per-job stage timings in microseconds (when measured).
///
/// On CUDA, `h2d_us` / `kernel_us` / `d2h_us` come from `CudaEvent` elapsed times
/// recorded around H2D, kernel launch, and optional D2H for reductions.
/// `queue_wait_us` is wall-clock time spent in the scheduler queue before issue.
#[derive(Debug, Clone, Copy, Default)]
pub struct JobMetrics {
    /// Micros from enqueue until the job was issued to a backend slot.
    pub queue_wait_us: u64,
    /// Host→device transfer stage (CUDA events); 0 on host backend.
    pub h2d_us: u64,
    /// Kernel / compute stage (CUDA events); 0 on host backend.
    pub kernel_us: u64,
    /// Device→host stage when applicable (e.g. reduction partials); else 0.
    pub d2h_us: u64,
}

/// Aggregate counters for one runtime instance.
///
/// Useful for benches (`pool_hits` vs `pool_misses`, `last_gpu_ms`) and for
/// asserting that contract tests exercise the intended code paths.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeMetrics {
    /// Jobs accepted by `submit` (includes ones that later fail).
    pub jobs_submitted: u64,
    /// Jobs that reached Completed successfully.
    pub jobs_completed: u64,
    /// Jobs that reached Failed (issue error or backend failure).
    pub jobs_failed: u64,
    /// Cumulative bytes copied host→device on CUDA.
    pub bytes_h2d: u64,
    /// Cumulative bytes copied device→host on CUDA.
    pub bytes_d2h: u64,
    /// Number of times `poll()` was entered.
    pub polls: u64,
    /// Host or device pool freelist hits.
    pub pool_hits: u64,
    /// Fresh allocations that missed the freelist.
    pub pool_misses: u64,
    /// Last measured end-to-end GPU stage time in milliseconds (CUDA only).
    /// Host backend never writes a fake value here.
    pub last_gpu_ms: f64,
}
