//! Pack AI — pré-processamento de tensores/imagens.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

pub mod prelude {
    pub use crate::{normalize, NormalizeConfig};
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizeConfig {
    pub eps: f32,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self { eps: 1e-5 }
    }
}

/// Normaliza com mean/std estimados no host (0.1). Depois: reduction GPU.
pub fn normalize(
    device: &mut Device,
    input: &Buffer,
    output: &Buffer,
    cfg: NormalizeConfig,
) -> Result<Submission> {
    let data = device.download(input)?;
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|v| {
        let d = v - mean;
        d * d
    }).sum::<f32>()
        / n;
    let inv_std = 1.0 / (var.sqrt() + cfg.eps);
    device.submit(Operation::Normalize {
        input: input.id(),
        output: output.id(),
        mean,
        inv_std,
    })
}
