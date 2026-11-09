//! # corewarp-game
//!
//! Domain pack for game / simulation workloads on the shared CoreWarp runtime.
//!
//! Enable with `--features game` on the facade crate. 0.1 ships a single particle
//! integrator that mutates an Array-of-Structures (AoS) state buffer in place.
//!
//! ## Particle layout
//! Each particle occupies 6 consecutive `f32` values:
//! `[px, py, pz, vx, vy, vz]`. A buffer of `N` particles therefore has length `6N`.
//! The runtime shape validator rejects lengths that are not multiples of 6.

use corewarp_core::{Operation, Result};
use corewarp_runtime::{Buffer, Device, Submission};

/// Re-exports commonly used by game examples (`use corewarp::game::prelude::*`).
pub mod prelude {
    pub use crate::{update_particles, ParticleUpdateConfig};
}

/// Integration parameters for [`update_particles`].
#[derive(Debug, Clone, Copy)]
pub struct ParticleUpdateConfig {
    /// Simulation timestep in seconds (default `1/60`).
    pub delta_time: f32,
    /// Constant gravity applied to `vy` each step (default `-9.81`).
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

/// Submit one Euler integration step over the AoS particle `state` buffer.
///
/// For each particle `i`:
/// 1. `vy += gravity_y * delta_time`
/// 2. `px += vx * delta_time`, `py += vy * delta_time`, `pz += vz * delta_time`
///
/// The update is **in-place** (`state` is both read and written), so overlapping
/// submits on the same buffer are serialized by the runtime conflict locks.
/// Returns a [`Submission`]; call `device.wait` / `drain` before reading results.
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
