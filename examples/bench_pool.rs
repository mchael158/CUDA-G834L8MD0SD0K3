//! Experiment 2: alloc-per-job CUDA baseline vs CoreWarp resident device pool.
//!
//! Contrasts a path that allocates/frees device memory every job against the
//! runtime's pooled resident buffers. Requires `--features cuda`.
//! Run with: `cargo run -p corewarp --release --example bench_pool --features cuda`

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

/// Manual path: fresh device alloc + H2D for x and y on every job (drops free on scope exit).
fn alloc_per_job(n: usize, jobs: usize) -> Result<(f64, u64), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| format!("{e:?}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("{e:?}"))?;
    let f = module
        .load_function("saxpy")
        .map_err(|e| format!("{e:?}"))?;

    let x_host = vec![1.0f32; n];
    let mut allocs = 0u64;
    let t0 = Instant::now();
    for _ in 0..jobs {
        let y_host = vec![2.0f32; n];
        let d_x = stream.clone_htod(&x_host).map_err(|e| format!("{e:?}"))?;
        allocs += 1;
        let mut d_y = stream.clone_htod(&y_host).map_err(|e| format!("{e:?}"))?;
        allocs += 1;
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
        let _ = stream.clone_dtoh(&d_y).map_err(|e| format!("{e:?}"))?;
        // d_x / d_y drop → device free
    }
    stream.synchronize().map_err(|e| format!("{e:?}"))?;
    Ok((t0.elapsed().as_secs_f64(), allocs))
}

/// Runtime path: pre-upload a ring of y buffers and reuse residents across rounds.
fn runtime_pool(n: usize, jobs: usize) -> corewarp::Result<(f64, u64, u64)> {
    let mut rt = CoreWarp::builder()
        .backend(BackendKind::Cuda)
        .require_cuda(true)
        .streams(2)
        .issue_width(2)
        .max_in_flight(64)
        .scheduling(SchedulingPolicy::Fixed)
        .build()?;

    let x = rt.upload(&vec![1.0; n])?;
    // Pre-allocate ys once; reuse resident device buffers across submit rounds.
    let ys: Vec<_> = (0..jobs.min(32))
        .map(|_| rt.upload(&vec![2.0; n]))
        .collect::<Result<Vec<_>, _>>()?;

    let t0 = Instant::now();
    for i in 0..jobs {
        let y = ys[i % ys.len()];
        rt.submit(Operation::Saxpy { alpha: 3.0, x, y })?;
        if (i + 1) % ys.len() == 0 {
            rt.drain()?;
        }
    }
    rt.drain()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let m = rt.metrics();
    rt.shutdown()?;
    Ok((elapsed, m.pool_hits, m.pool_misses))
}

fn main() {
    let n = 1 << 14;
    let jobs = 512;
    println!("mode,n,jobs,elapsed_ms,extra");
    match alloc_per_job(n, jobs) {
        Ok((elapsed, allocs)) => println!(
            "alloc_per_job,{n},{jobs},{:.3},device_allocs≈{allocs}",
            elapsed * 1000.0
        ),
        Err(e) => {
            eprintln!("alloc_per_job failed: {e}");
            std::process::exit(1);
        }
    }
    match runtime_pool(n, jobs) {
        Ok((elapsed, hits, misses)) => println!(
            "runtime_pool,{n},{jobs},{:.3},pool_hits={hits};pool_misses={misses}",
            elapsed * 1000.0
        ),
        Err(e) => {
            eprintln!("runtime_pool failed: {e}");
            std::process::exit(1);
        }
    }
}
