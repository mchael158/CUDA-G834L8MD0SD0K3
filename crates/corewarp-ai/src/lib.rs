//! # corewarp-ai
//!
//! Domain pack for AI / tensor preprocessing built on the shared CoreWarp runtime.
//!
//! Enable with `--features ai` on the facade crate. 0.1 focuses on a practical
//! normalize path used by image/tensor examples: estimate mean and variance on the
//! **host**, then dispatch a GPU/host [`Operation::Normalize`] that applies
//! `(x - mean) * inv_std` element-wise.
//!
//! Future iterations may move the statistics reductions entirely onto the device;
//! the public `normalize` signature is intentionally stable for that migration.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

/// Re-exports commonly used by AI examples (`use corewarp::ai::prelude::*`).
pub mod prelude {
    pub use crate::{normalize, NormalizeConfig};
}

/// Tunables for [`normalize`].
#[derive(Debug, Clone, Copy)]
pub struct NormalizeConfig {
    /// Epsilon added to `sqrt(variance)` before reciprocal to avoid division by zero
    /// on constant tensors. Default `1e-5`.
    pub eps: f32,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self { eps: 1e-5 }
    }
}

/// Download `input`, estimate population mean/variance on the host, then submit
/// a device [`Operation::Normalize`] that writes the standardized values to `output`.
///
/// ## Algorithm (0.1)
/// 1. `device.download(input)` — may block until any prior writers finish and the
///    host mirror is coherent.
/// 2. `mean = Σx / n`, `var = Σ(x-mean)² / n` (population variance).
/// 3. `inv_std = 1 / (sqrt(var) + eps)`.
/// 4. Submit normalize kernel/op; return the [`Submission`] without waiting.
///
/// ## Notes
/// The statistics step is host-side by design in 0.1 so both CUDA and host backends
/// share identical numeric behavior. Prefer calling this once per tensor rather than
/// inside a tight loop if download cost dominates.
pub fn normalize(
    device: &mut Device,
    input: &Buffer,
    output: &Buffer,
    cfg: NormalizeConfig,
) -> Result<Submission> {
    let data = device.download(input)?;
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (var.sqrt() + cfg.eps);
    device.submit(Operation::Normalize {
        input: input.id(),
        output: output.id(),
        mean,
        inv_std,
    })
}
