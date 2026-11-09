//! Exemplo de domínio imagem: scale + histogram + prefix_sum (host).
use corewarp::prelude::*;
use corewarp::{histogram, prefix_sum, vector_scale};

fn main() -> Result<()> {
    let mut device = Device::host()?;
    println!("backend: {}", device.backend_name());

    let pixels: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let input = device.upload(&pixels)?;
    let scaled = device.allocate(pixels.len())?;
    let bins = device.allocate(8)?;
    let cdf = device.allocate(8)?;

    let s1 = vector_scale(&mut device, &input, &scaled, 1.0)?;
    device.wait(&s1)?;
    let s2 = histogram(&mut device, &scaled, &bins, 0.0, 64.0)?;
    device.wait(&s2)?;
    let s3 = prefix_sum(&mut device, &bins, &cdf)?;
    device.wait(&s3)?;

    let hist = device.download(&bins)?;
    let cdf_v = device.download(&cdf)?;
    println!("image_processing ok hist[0]={:.0} cdf_last={:.0}", hist[0], cdf_v[7]);
    device.shutdown()?;
    Ok(())
}
