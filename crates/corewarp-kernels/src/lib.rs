//! Operações compartilhadas de alto nível.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

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

pub fn reduction_sum(device: &mut Device, input: &Buffer, output: &Buffer) -> Result<Submission> {
    device.submit(Operation::ReductionSum {
        input: input.id(),
        output: output.id(),
    })
}

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

pub fn saxpy(device: &mut Device, x: &Buffer, y: &Buffer, alpha: f32) -> Result<Submission> {
    device.submit(Operation::Saxpy {
        alpha,
        x: x.id(),
        y: y.id(),
    })
}

/// Scan inclusivo: `out[i] = sum(in[0..=i])`.
pub fn prefix_sum(device: &mut Device, input: &Buffer, output: &Buffer) -> Result<Submission> {
    device.submit(Operation::PrefixSum {
        input: input.id(),
        output: output.id(),
    })
}

/// Histograma em `output.len()` bins sobre `[min_v, max_v]`.
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
