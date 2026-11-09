//! Core value types: IDs, scheduling, configuration, and typed GPU/host operations.
//!
//! Operations are *descriptions* of work. They do not execute themselves; the
//! runtime validates shapes, tracks buffer conflicts, and dispatches to a backend.

use crate::error::{Error, Result};

/// Opaque identity of a queued/submitted job inside one [`crate`] runtime instance.
///
/// IDs are allocated monotonically starting at 1 for a given runtime. They are not
/// globally unique across processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

/// Opaque identity of a buffer owned by the memory manager.
///
/// After [`corewarp_memory`] frees a buffer, reusing the same numeric ID is undefined;
/// always treat handles returned by upload/allocate as the source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

/// User-facing handle returned by `submit` / `submit_with_priority`.
///
/// Wrap a [`JobId`] so callers poll/wait without poking the scheduler queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobHandle {
    pub id: JobId,
}

/// Relative scheduling priority for queued jobs.
///
/// Higher numeric values win when the scheduler picks the next job. Waiting jobs
/// may also receive an aging boost so Low work is not starved forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// Lifecycle of a job inside the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Accepted into the priority queue; not yet issued to a backend slot.
    Queued,
    /// Reserved for future “submitted but not running” semantics (0.1 uses Running).
    Submitted,
    /// Issued to a backend slot; completion event not yet observed.
    Running,
    /// Finished successfully; buffers released from in-flight tracking.
    Completed,
    /// Finished with an error recorded on the job record.
    Failed,
}

/// Coarse status exposed by the public `Device` / `Submission` API.
///
/// Maps from [`JobStatus`]: anything not terminal becomes [`Pending`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionStatus {
    Pending,
    Completed,
    Failed,
}

/// How the scheduler chooses concurrency over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Keep `active_streams == stream_count` for the whole run.
    Fixed,
    /// Periodically nudge active stream count based on observed queue depth
    /// (conservative heuristic; never fabricates GPU timing on host).
    Adaptive,
}

/// Which execution backend the runtime should prefer at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Pure host CPU reference backend (always available; validates contracts).
    Host,
    /// CUDA driver backend (requires `cuda` feature + working Toolkit/driver).
    Cuda,
}

/// Tunables applied when constructing a [`CoreWarp`](struct@RuntimeConfig) runtime.
///
/// Defaults target a small multi-stream desktop GPU setup. Host benches may raise
/// `memory_pool_bytes` to avoid OOM on Large size-class churn.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Logical stream/slot count exposed to the backend.
    pub stream_count: usize,
    /// Hard cap on concurrently in-flight backend jobs.
    pub max_in_flight: usize,
    /// Max jobs the poller attempts to issue per `poll()` call.
    pub issue_width: usize,
    /// Host memory pool budget in bytes (size-class allocations charge against this).
    pub memory_pool_bytes: usize,
    pub scheduling: SchedulingPolicy,
    pub backend: BackendKind,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            stream_count: 4,
            max_in_flight: 32,
            issue_width: 2,
            memory_pool_bytes: 128 * 1024 * 1024,
            scheduling: SchedulingPolicy::Fixed,
            // Prefer CUDA when the facade asks for it; host override is explicit.
            backend: BackendKind::Cuda,
        }
    }
}

/// Human-readable description of the opened device for logging/UI.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Static backend tag: `"host"` or `"cuda"`.
    pub backend: &'static str,
    /// Device ordinal (0.1 only supports CUDA index 0).
    pub index: u32,
    /// Display name (`"HostCPU"`, `"CUDA:0"`, …).
    pub name: String,
}

/// Typed GPU/host workloads identified by buffer handles.
///
/// Each variant lists the buffers it reads and writes. The runtime uses
/// [`Operation::reads`] / [`Operation::writes`] for conflict locks and
/// [`Operation::validate_shapes`] before enqueue.
#[derive(Debug, Clone)]
pub enum Operation {
    /// BLAS-like SAXPY: `y[i] += alpha * x[i]` (in-place on `y`).
    Saxpy {
        alpha: f32,
        x: BufferId,
        y: BufferId,
    },
    /// Element-wise sum: `output[i] = a[i] + b[i]` (all three same length).
    VectorAdd {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },
    /// Element-wise scale: `output[i] = alpha * input[i]`.
    VectorScale {
        alpha: f32,
        input: BufferId,
        output: BufferId,
    },
    /// Full reduction: `output[0] = sum(input[*])`. Output length must be exactly 1.
    ReductionSum {
        input: BufferId,
        output: BufferId,
    },
    /// Affine normalize: `out[i] = (in[i] - mean) * inv_std`.
    ///
    /// Mean / inv_std are supplied by the caller (AI pack currently estimates them
    /// on the host). This keeps the kernel itself a pure element-wise transform.
    Normalize {
        input: BufferId,
        output: BufferId,
        mean: f32,
        inv_std: f32,
    },
    /// In-place Euler integration over AoS particles packed as
    /// `[px, py, pz, vx, vy, vz] * N` (length must be a multiple of 6).
    ///
    /// Each step: `vy += gravity_y * dt`, then `p += v * dt`.
    ParticleUpdate {
        state: BufferId,
        delta_time: f32,
        gravity_y: f32,
    },
    /// Inclusive prefix sum (scan): `out[i] = sum(in[0..=i])`.
    ///
    /// Input and output must share the same length. CUDA 0.1 supports `n <= 65536`
    /// (256 blocks × 256 threads); larger `n` returns [`Error::Invalid`].
    PrefixSum {
        input: BufferId,
        output: BufferId,
    },
    /// Histogram: count samples from `input` into `output.len()` bins spanning
    /// the half-open value range `[min_v, max_v)`.
    ///
    /// Values are clamped into the range; the uppermost edge maps to the last bin
    /// via a `0.999999` clamp so `max_v` itself does not overflow the bin index.
    Histogram {
        input: BufferId,
        output: BufferId,
        min_v: f32,
        max_v: f32,
    },
}

impl Operation {
    /// Union of read and write buffer IDs (deduplicated, reads first).
    ///
    /// Used when retaining/releasing in-flight refcounts on the memory manager.
    pub fn buffer_ids(&self) -> Vec<BufferId> {
        let mut ids = self.reads();
        for w in self.writes() {
            if !ids.contains(&w) {
                ids.push(w);
            }
        }
        ids
    }

    /// Buffers that must be coherent before the op starts (may also be written).
    pub fn reads(&self) -> Vec<BufferId> {
        match *self {
            Self::Saxpy { x, y, .. } => {
                // SAXPY reads both x and y; if they alias, report once.
                if x == y {
                    vec![x]
                } else {
                    vec![x, y]
                }
            }
            Self::VectorAdd { a, b, .. } => vec![a, b],
            Self::VectorScale { input, .. } => vec![input],
            Self::ReductionSum { input, .. } => vec![input],
            Self::Normalize { input, .. } => vec![input],
            Self::ParticleUpdate { state, .. } => vec![state],
            Self::PrefixSum { input, .. } => vec![input],
            Self::Histogram { input, .. } => vec![input],
        }
    }

    /// Buffers mutated by the op (exclusive-write conflict domain).
    pub fn writes(&self) -> Vec<BufferId> {
        match *self {
            Self::Saxpy { y, .. } => vec![y],
            Self::VectorAdd { output, .. } => vec![output],
            Self::VectorScale { output, .. } => vec![output],
            Self::ReductionSum { output, .. } => vec![output],
            Self::Normalize { output, .. } => vec![output],
            Self::ParticleUpdate { state, .. } => vec![state],
            Self::PrefixSum { output, .. } => vec![output],
            Self::Histogram { output, .. } => vec![output],
        }
    }

    /// Classic reader/writer hazard check against currently in-flight buffers.
    ///
    /// Returns true if issuing `self` now could race with active work:
    /// - a write of ours overlaps any active read or write, or
    /// - a read of ours overlaps any active write.
    pub fn conflicts_with(&self, active_reads: &[BufferId], active_writes: &[BufferId]) -> bool {
        for w in self.writes() {
            if active_writes.contains(&w) || active_reads.contains(&w) {
                return true;
            }
        }
        for r in self.reads() {
            if active_writes.contains(&r) {
                return true;
            }
        }
        false
    }

    /// Enforce per-op length / range contracts before enqueue.
    ///
    /// `len_of` is supplied by the runtime so this crate stays free of the memory
    /// manager dependency. Empty inputs and mismatched lengths return [`Error::Invalid`].
    pub fn validate_shapes(&self, len_of: impl Fn(BufferId) -> Result<usize>) -> Result<()> {
        match *self {
            Self::Saxpy { x, y, .. } => {
                let nx = len_of(x)?;
                let ny = len_of(y)?;
                if nx == 0 || nx != ny {
                    return Err(Error::Invalid("saxpy: incompatible x/y lengths"));
                }
                // Distinct buffers required: CUDA take_resident would orphan the first
                // resident if x == y, leaving host_dirty cleared and the next job wrong.
                if x == y {
                    return Err(Error::Invalid("saxpy: x and y must be distinct buffers"));
                }
            }
            Self::VectorAdd { a, b, output } => {
                let na = len_of(a)?;
                let nb = len_of(b)?;
                let no = len_of(output)?;
                if na == 0 || na != nb || na != no {
                    return Err(Error::Invalid("vector_add: incompatible lengths"));
                }
                // Pairwise-distinct operands: double take_resident on the same id
                // drops the resident map entry and breaks later H2D coherence.
                if a == b || a == output || b == output {
                    return Err(Error::Invalid(
                        "vector_add: a, b, and output must be distinct buffers",
                    ));
                }
            }
            Self::VectorScale { input, output, .. } | Self::Normalize { input, output, .. } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 || ni != no {
                    return Err(Error::Invalid("scale/normalize: incompatible lengths"));
                }
                if input == output {
                    return Err(Error::Invalid(
                        "scale/normalize: input and output must be distinct buffers",
                    ));
                }
            }
            Self::ReductionSum { input, output } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 {
                    return Err(Error::Invalid("reduction: empty input"));
                }
                if no != 1 {
                    return Err(Error::Invalid("reduction: output length must be 1"));
                }
                if input == output {
                    return Err(Error::Invalid(
                        "reduction: input and output must be distinct buffers",
                    ));
                }
            }
            Self::ParticleUpdate { state, .. } => {
                let n = len_of(state)?;
                if n == 0 || n % 6 != 0 {
                    return Err(Error::Invalid(
                        "particle: length must be a non-zero multiple of 6",
                    ));
                }
            }
            Self::PrefixSum { input, output } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 || ni != no {
                    return Err(Error::Invalid("prefix_sum: incompatible lengths"));
                }
                if input == output {
                    return Err(Error::Invalid(
                        "prefix_sum: input and output must be distinct buffers",
                    ));
                }
            }
            Self::Histogram {
                input,
                output,
                min_v,
                max_v,
            } => {
                let ni = len_of(input)?;
                let bins = len_of(output)?;
                if ni == 0 || bins == 0 {
                    return Err(Error::Invalid("histogram: empty input or bins"));
                }
                if !(max_v > min_v) {
                    return Err(Error::Invalid("histogram: max_v must be greater than min_v"));
                }
                if input == output {
                    return Err(Error::Invalid(
                        "histogram: input and output must be distinct buffers",
                    ));
                }
            }
        }
        Ok(())
    }
}
