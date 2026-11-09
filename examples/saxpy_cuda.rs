//! CUDA SAXPY smoke test: large vector, resident reuse, event-based job metrics.
//!
//! Requires `--features cuda` and a working CUDA Toolkit/driver.
//! Run with: `cargo run -p corewarp --release --example saxpy_cuda --features cuda`
//!
//! Expected numeric result after two submits on the same `y`:
//! 1st: y = 3*1 + 2 = 5; 2nd: y = 1*1 + 5 = 6.

use corewarp::{BackendKind, CoreWarp, Operation, Priority, SchedulingPolicy};

fn main() -> corewarp::Result<()> {
    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Cuda)
        .require_cuda(true)
        .streams(4)
        .issue_width(2)
        .max_in_flight(32)
        .scheduling(SchedulingPolicy::Fixed)
        .build()?;

    println!("backend: {}", rt.backend_name());

    const N: usize = 1 << 22;
    let x = rt.upload(&vec![1.0; N])?;
    let y = rt.upload(&vec![2.0; N])?;

    let handle = rt.submit_with_priority(
        Priority::High,
        Operation::Saxpy {
            alpha: 3.0,
            x,
            y,
        },
    )?;
    rt.wait(&handle)?;

    // Second submit on the same y: should reuse the resident device copies
    // (no H2D of x if it is still clean / not host_dirty).
    let handle2 = rt.submit(Operation::Saxpy {
        alpha: 1.0,
        x,
        y,
    })?;
    rt.wait(&handle2)?;

    let y_host = rt.download(y)?;
    let jm = rt.job_metrics(&handle)?;
    let m = rt.metrics();
    println!(
        "submitted={} completed={} h2d={} d2h={} gpu_ms={:.4} job0_h2d_us={} job0_kernel_us={} y[0]={:.1}",
        m.jobs_submitted,
        m.jobs_completed,
        m.bytes_h2d,
        m.bytes_d2h,
        m.last_gpu_ms,
        jm.h2d_us,
        jm.kernel_us,
        y_host[0]
    );

    // 1st: y=3*1+2=5; 2nd: y=1*1+5=6
    if (y_host[0] - 6.0).abs() > 0.1 {
        eprintln!("unexpected y[0]={}", y_host[0]);
        std::process::exit(2);
    }

    rt.shutdown()?;
    println!("saxpy_cuda ok");
    Ok(())
}
