//! AI-pack example: host-estimated mean/std followed by a Normalize op.
//!
//! Run with: `cargo run -p corewarp --release --example tensor_operations --features ai`

use corewarp::ai::prelude::*;
use corewarp::prelude::*;

fn main() -> Result<()> {
    let mut device = Device::host()?;
    let input = device.upload(&[1.0, 2.0, 3.0, 4.0])?;
    let output = device.allocate(4)?;

    // normalize() downloads input, computes mean/var on the host, then submits Normalize.
    let sub = normalize(&mut device, &input, &output, NormalizeConfig::default())?;
    device.wait(&sub)?;
    let out = device.download(&output)?;
    println!("tensor_operations ok out={out:?}");
    device.shutdown()?;
    Ok(())
}
