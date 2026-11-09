//! Experiment 1: raw multi-stream CUDA launches vs CoreWarp (requires `--features cuda`).
//!
//! Prints CSV rows comparing wall time for the same SAXPY workload under a
//! hand-rolled stream pool versus the CoreWarp runtime. Run with:
//! `cargo run -p corewarp --release --example bench_manual_vs_runtime --features cuda`

use std::time::Instant;

use corewarp::{BackendKind, CoreWarp, Operation, SchedulingPolicy};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNEL: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

/// Baseline: one H2D+kernel+D2H per job, round-robin across `streams` CUDA streams.
fn manual(n: usize, jobs: usize, streams: usize) -> Result<(f64, f32), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
    let ptx = compile_ptx(KERNEL).map_err(|e| format!("{e:?}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("{e:?}"))?;
    let f = module
        .load_function("saxpy")
        .map_err(|e| format!("{e:?}"))?;

    let mut stream_pool = Vec::new();
    for _ in 0..streams {
        stream_pool.push(ctx.new_stream().map_err(|e| format!("{e:?}"))?);
    }

    let x_host = vec![1.0f32; n];
    let mut y_hosts: Vec<Vec<f32>> = (0..jobs).map(|_| vec![2.0f32; n]).collect();

    let t0 = Instant::now();
    for (i, y_host) in y_hosts.iter_mut().enumerate() {
        let stream = &stream_pool[i % streams];
        let d_x = stream.clone_htod(&x_host).map_err(|e| format!("{e:?}"))?;
        let mut d_y = stream.clone_htod(y_host).map_err(|e| format!("{e:?}"))?;
        let alpha = 3.0f32;
        let n_arg = n;
        let mut b = stream.launch_builder(&f);
        b.arg(&alpha);
        b.arg(&d_x);
        b.arg(&mut d_y);
        b.arg(&n_arg);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((n as u32) + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("{e:?}"))?;
        stream.memcpy_dtoh(&d_y, y_host).map_err(|e| format!("{e:?}"))?;
    }
    for s in &stream_pool {
        s.synchronize().map_err(|e| format!("{e:?}"))?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, y_hosts[0][0]))
}

/// CoreWarp path: upload once, submit many SAXPYs, drain, download one result.
fn runtime(n: usize, jobs: usize, streams: usize) -> corewarp::Result<(f64, f32)> {
    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Cuda)
        .require_cuda(true)
        .streams(streams)
        .issue_width(streams.min(4))
        .max_in_flight(jobs.max(32))
        .scheduling(SchedulingPolicy::Fixed)
        .build()?;

    let x = rt.upload(&vec![1.0; n])?;
    let mut ys = Vec::with_capacity(jobs);
    for _ in 0..jobs {
        ys.push(rt.upload(&vec![2.0; n])?);
    }

    let t0 = Instant::now();
    for &y in &ys {
        rt.submit(Operation::Saxpy { alpha: 3.0, x, y })?;
    }
    rt.drain()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let y0 = rt.download(ys[0])?[0];
    rt.shutdown()?;
    Ok((elapsed, y0))
}

fn main() {
    let cases = [
        (1 << 12, 128usize, 2usize),
        (1 << 16, 128, 4),
        (1 << 20, 64, 4),
    ];

    println!("mode,n,jobs,streams,elapsed_ms,jobs_per_s,y0");
    for &(n, jobs, streams) in &cases {
        match manual(n, jobs, streams) {
            Ok((elapsed, y0)) => println!(
                "manual,{n},{jobs},{streams},{:.3},{:.1},{y0}",
                elapsed * 1000.0,
                jobs as f64 / elapsed
            ),
            Err(e) => {
                eprintln!("manual failed: {e}");
                std::process::exit(1);
            }
        }
        match runtime(n, jobs, streams) {
            Ok((elapsed, y0)) => println!(
                "runtime,{n},{jobs},{streams},{:.3},{:.1},{y0}",
                elapsed * 1000.0,
                jobs as f64 / elapsed
            ),
            Err(e) => {
                eprintln!("runtime failed: {e}");
                std::process::exit(1);
            }
        }
    }
}
