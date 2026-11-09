//! Game-pack example: one Euler particle step on the host backend.
//!
//! Layout reminder: each particle is six floats `[px, py, pz, vx, vy, vz]`.
//! Run with: `cargo run -p corewarp --release --example particle_simulation --features game`

use corewarp::game::prelude::*;
use corewarp::prelude::*;

fn main() -> Result<()> {
    let mut device = Device::host()?;
    // Two particles packed AoS: px py pz vx vy vz
    let state = device.upload(&[
        0.0, 10.0, 0.0, 1.0, 0.0, 0.0, // particle 0
        0.0, 5.0, 0.0, -1.0, 2.0, 0.0, // particle 1
    ])?;

    let sub = update_particles(
        &mut device,
        &state,
        ParticleUpdateConfig {
            delta_time: 0.016,
            gravity_y: -9.81,
        },
    )?;
    device.wait(&sub)?;
    let out = device.download(&state)?;
    println!(
        "particle_simulation ok y0={:.4} vy0={:.4}",
        out[1], out[4]
    );
    device.shutdown()?;
    Ok(())
}
