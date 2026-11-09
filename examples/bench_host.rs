//! Host throughput bench: scheduler + pool without CUDA (always runnable).
//!
//! Measures jobs/s for a wave of SAXPYs, then repeats after free+re-upload to
//! show pool reuse (pool_hits). Run with:
//! `cargo run -p corewarp --release --example bench_host`

use std::time::Instant;

use corewarp::{BackendKind, CoreWarp, Operation, SchedulingPolicy};

fn main() -> corewarp::Result<()> {
    const N: usize = 1 << 12; // 16 KiB → Medium size class
    const JOBS: usize = 256;

    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Host)
        .streams(4)
        .issue_width(2)
        .max_in_flight(32)
        .memory_pool_bytes(256 * 1024 * 1024)
        .scheduling(SchedulingPolicy::Fixed)
        .build()?;

    let x = rt.upload(&vec![1.0; N])?;
    let mut ys = Vec::new();
    for _ in 0..JOBS {
        ys.push(rt.upload(&vec![2.0; N])?);
    }

    let t0 = Instant::now();
    for &y in &ys {
        rt.submit(Operation::Saxpy { alpha: 3.0, x, y })?;
    }
    rt.drain()?;
    let elapsed = t0.elapsed();

    let m = rt.metrics();
    let jobs_per_s = JOBS as f64 / elapsed.as_secs_f64();
    println!(
        "bench_host N={N} jobs={JOBS} elapsed={:.3}ms jobs/s={:.0} completed={} pool_hits={} pool_misses={}",
        elapsed.as_secs_f64() * 1000.0,
        jobs_per_s,
        m.jobs_completed,
        m.pool_hits,
        m.pool_misses
    );

    // Reuse path: free every y, then re-upload + submit again (expect pool hits).
    for y in ys {
        rt.free_buffer(y)?;
    }
    let t1 = Instant::now();
    for _ in 0..JOBS {
        let y = rt.upload(&vec![2.0; N])?;
        rt.submit(Operation::Saxpy { alpha: 3.0, x, y })?;
    }
    rt.drain()?;
    let elapsed2 = t1.elapsed();
    let m2 = rt.metrics();
    println!(
        "bench_host_reuse elapsed={:.3}ms jobs/s={:.0} pool_hits={} pool_misses={}",
        elapsed2.as_secs_f64() * 1000.0,
        JOBS as f64 / elapsed2.as_secs_f64(),
        m2.pool_hits,
        m2.pool_misses
    );

    rt.shutdown()?;
    Ok(())
}
