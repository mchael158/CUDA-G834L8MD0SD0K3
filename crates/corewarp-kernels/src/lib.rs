//! # corewarp-kernels
//!
//! Thin, shared high-level wrappers around [`corewarp_core::Operation`] variants.
//!
//! These helpers exist so AI/Game packs and examples can call named functions
//! (`saxpy`, `prefix_sum`, …) without constructing enum variants by hand. Each
//! function simply builds the corresponding [`Operation`] and forwards to
//! [`Device::submit`](corewarp_runtime::Device::submit). No extra buffering or
//! synchronization is performed here — callers own `wait` / `drain`.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

/// Submit `output[i] = alpha * input[i]` (element-wise scale).
///
/// Requires `input.len() == output.len() > 0`. Returns a [`Submission`] handle;
/// the Device does not block until you `wait` or `drain`.
pub fn vector_scale(
    device: &mut Device,
    input: &Buffer,
    output: &Buffer,
    alpha: f32,
) -> Result<Submission> {
    device.submit(Operation::VectorScale {
        alpha,
        input: input.id(),
        output: output.id(),
    })
}

/// Submit a full sum reduction into a length-1 `output` buffer.
///
/// On CUDA this is a block-reduce + host-side partial sum after the completion
/// event; on host it is a straight iterator sum. Semantics: `output[0] = Σ input`.
pub fn reduction_sum(device: &mut Device, input: &Buffer, output: &Buffer) -> Result<Submission> {
    device.submit(Operation::ReductionSum {
        input: input.id(),
        output: output.id(),
    })
}

/// Submit `output[i] = a[i] + b[i]` for equal-length buffers.
pub fn vector_add(
    device: &mut Device,
    a: &Buffer,
    b: &Buffer,
    output: &Buffer,
) -> Result<Submission> {
    device.submit(Operation::VectorAdd {
        a: a.id(),
        b: b.id(),
        output: output.id(),
    })
}

/// Submit in-place SAXPY: `y[i] += alpha * x[i]`.
///
/// `x` and `y` must have identical non-zero lengths. Concurrent SAXPYs that share
/// the same `y` are serialized by the runtime conflict locks.
pub fn saxpy(device: &mut Device, x: &Buffer, y: &Buffer, alpha: f32) -> Result<Submission> {
    device.submit(Operation::Saxpy {
        alpha,
        x: x.id(),
        y: y.id(),
    })
}

/// Submit an **inclusive** prefix-sum (scan): `out[i] = sum(in[0..=i])`.
///
/// Input and output must be the same length. CUDA 0.1 supports at most 65536
/// elements (two-level block scan). Larger inputs fail validation / issue.
pub fn prefix_sum(device: &mut Device, input: &Buffer, output: &Buffer) -> Result<Submission> {
    device.submit(Operation::PrefixSum {
        input: input.id(),
        output: output.id(),
    })
}

/// Submit a histogram over `input` into `output.len()` bins covering `[min_v, max_v)`.
///
/// Bin index = clamp((v - min_v) / (max_v - min_v)) * bins. CUDA uses `atomicAdd`
/// into the bin buffer after zero-filling it; host uses a scalar loop.
pub fn histogram(
    device: &mut Device,
    input: &Buffer,
    output: &Buffer,
    min_v: f32,
    max_v: f32,
) -> Result<Submission> {
    device.submit(Operation::Histogram {
        input: input.id(),
        output: output.id(),
        min_v,
        max_v,
    })
}
