//! Pack Game — simulação e processamento para jogos.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

pub mod prelude {
    pub use crate::{update_particles, ParticleUpdateConfig};
}

#[derive(Debug, Clone, Copy)]
pub struct ParticleUpdateConfig {
    pub delta_time: f32,
    pub gravity_y: f32,
}

impl Default for ParticleUpdateConfig {
    fn default() -> Self {
        Self {
            delta_time: 1.0 / 60.0,
            gravity_y: -9.81,
        }
    }
}

/// Atualiza partículas AoS `[px,py,pz,vx,vy,vz] * N`.
pub fn update_particles(
    device: &mut Device,
    state: &Buffer,
    cfg: ParticleUpdateConfig,
) -> Result<Submission> {
    device.submit(Operation::ParticleUpdate {
        state: state.id(),
        delta_time: cfg.delta_time,
        gravity_y: cfg.gravity_y,
    })
}
