//! Exemplo AI: normalize.
use corewarp::ai::prelude::*;
use corewarp::prelude::*;

fn main() -> Result<()> {
    let mut device = Device::host()?;
    let input = device.upload(&[1.0, 2.0, 3.0, 4.0])?;
    let output = device.allocate(4)?;

    let sub = normalize(&mut device, &input, &output, NormalizeConfig::default())?;
    device.wait(&sub)?;
    let out = device.download(&output)?;
    println!("tensor_operations ok out={out:?}");
    device.shutdown()?;
    Ok(())
}
