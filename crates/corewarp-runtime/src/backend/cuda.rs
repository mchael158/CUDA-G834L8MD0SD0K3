//! CUDA backend for CoreWarp: device execution, memory residency, and staged timing.
//!
//! # Architecture overview
//!
//! This module implements the [`Backend`] trait using NVIDIA CUDA via `cudarc`. Work is issued
//! onto independent CUDA streams (one per scheduling slot) so multiple jobs can overlap in flight.
//!
//! ## Resident device buffers
//!
//! Each logical host buffer ([`BufferId`]) that participates in GPU operations may have a matching
//! [`Resident`] entry in `CudaBackend::resident`: a [`CudaSlice<f32>`] on the device whose logical
//! length is tracked separately from the pooled allocation size. Residents persist across jobs until
//! [`Backend::free_device`] or shutdown returns the slice to the pool. Kernels read/write through
//! `take_resident` / `put_resident` so the hash map can be temporarily bypassed without cloning
//! device memory.
//!
//! ## Pinned staging (H2D path)
//!
//! Host-to-device uploads do not call `memcpy_htod` directly from the CPU heap backing
//! [`MemoryManager`]. Instead, each slot owns an optional [`PinnedHostSlice<f32>`] sized with
//! [`pool_key`]: the next power of two of the element count, floored at 256 elements. The host
//! vector is copied into the pinned buffer (padding the tail with zeros), then the full pinned
//! region is copied to the resident device slice. This keeps H2D transfers page-locked and aligns
//! staging size with device pool keys.
//!
//! ## Dirty-flag driven H2D
//!
//! [`MemoryManager`] tracks `host_dirty` per buffer. [`ensure_resident`] allocates a resident slice
//! on first use, then runs [`h2d_buffer`] only when the buffer is newly resident or `host_dirty` is
//! set. After a successful upload, `host_dirty` is cleared. Device-side writes mark outputs via
//! `written` and `finish_slot` sets `device_dirty` on the host metadata when appropriate.
//!
//! ## Device pool (`next_power_of_two` keys)
//!
//! [`DevicePool`] reuses freed [`CudaSlice<f32>`] allocations keyed by [`pool_key(elems)`], which
//! is `elems.next_power_of_two().max(256)`. Acquire pops a slice of the requested key or allocates
//! zeros on miss; release pushes slices back after temporary buffers (e.g. prefix-sum block sums,
//! reduction partials) are done. Hit/miss counts feed [`RuntimeMetrics`].
//!
//! ## `CudaEvent` timing stages (h2d / kernel / d2h)
//!
//! Each issued job records [`StageEvents`]:
//! - **h2d**: from `h2d_start` through `h2d_end` (after all `ensure_resident` work on the stream).
//! - **kernel**: from `h2d_end` through `kernel_end` (after all kernel launches).
//! - **d2h**: optional; when reduction schedules a device-to-host of partial sums, `d2h_end` is
//!   recorded on the same stream so kernel time excludes that memcpy. Overall GPU wall time uses
//!   either `h2d_start`→`d2h_end` or `h2d_start`→`kernel_end`.
//!
//! Completion is tracked with a separate non-timing `done` event (`CU_EVENT_DISABLE_TIMING`).
//!
//! ## Prefix scan (two-level)
//!
//! [`Operation::PrefixSum`] runs three kernels in sequence (see Rust docs above each `const` / load):
//! 1. **`scan_blocks`**: per-block inclusive scan of input into output; per-block totals in `block_sums`.
//! 2. **`scan_small_inclusive`**: single-block inclusive scan over at most 256 block sums (limit `blocks <= 256`).
//! 3. **`add_block_offsets`**: add exclusive prefix of block totals into each element past block 0.
//!
//! Temporary `d_block` memory comes from the device pool and is released in `finish_slot`.
//!
//! ## Histogram
//!
//! [`Operation::Histogram`] first launches **`fill_zero`** to clear the output bin buffer, then
//! **`histogram`**, which maps each input value into `[min_v, max_v)`, clamps the normalized coordinate,
//! and accumulates with `atomicAdd` into bin counters (stored as `f32`).
//!
//! ## Reduction
//!
//! [`Operation::ReductionSum`] launches **`reduce_block`** to produce one partial sum per thread block.
//! Partials are copied to a **dedicated** pinned buffer (not the slot's H2D staging), and stored in
//! [`Slot::pending_reduce`]. The final scalar sum is computed on the **host** inside [`finish_slot`]
//! only after the job's `done` event reports completion—GPU time does not include that CPU reduction.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::backend::{Backend, CompletedWork};
use corewarp_core::{BufferId, JobId, Operation};
use corewarp_core::{Error, Result};
use corewarp_memory::MemoryManager;
use corewarp_telemetry::{JobMetrics, RuntimeMetrics};

/// Embedded CUDA C source compiled at backend initialization via NVRTC.
///
/// Kernel bodies must remain byte-identical to the shipped implementation; behavior is documented
/// in the module-level docs and in comments on the corresponding [`CudaFunction`] fields.
const KERNELS: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
extern "C" __global__ void vadd(const float* a, const float* b, float* o, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
extern "C" __global__ void vscale(float a, const float* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * in[i];
}
extern "C" __global__ void reduce_block(const float* in, float* partial, size_t n) {
    __shared__ float s[256];
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    s[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}
extern "C" __global__ void normalize(const float* in, float* out, float mean, float inv_std, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] - mean) * inv_std;
}
extern "C" __global__ void particle_update(float* state, float dt, float gravity_y, size_t n_particles) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        size_t b = i * 6;
        state[b + 4] += gravity_y * dt;
        state[b + 0] += state[b + 3] * dt;
        state[b + 1] += state[b + 4] * dt;
        state[b + 2] += state[b + 5] * dt;
    }
}
#define SCAN_BLOCK 256
extern "C" __global__ void scan_blocks(const float* in, float* out, float* block_sums, size_t n) {
    __shared__ float s[SCAN_BLOCK];
    size_t i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
        float add = 0.0f;
        if ((int)threadIdx.x >= offset) add = s[threadIdx.x - offset];
        __syncthreads();
        if ((int)threadIdx.x >= offset) s[threadIdx.x] += add;
        __syncthreads();
    }
    if (i < n) out[i] = s[threadIdx.x];
    size_t valid = n - blockIdx.x * SCAN_BLOCK;
    if (valid > SCAN_BLOCK) valid = SCAN_BLOCK;
    if (threadIdx.x + 1 == valid) block_sums[blockIdx.x] = s[threadIdx.x];
}
extern "C" __global__ void scan_small_inclusive(float* data, size_t n) {
    __shared__ float s[SCAN_BLOCK];
    s[threadIdx.x] = (threadIdx.x < n) ? data[threadIdx.x] : 0.0f;
    __syncthreads();
    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
        float add = 0.0f;
        if ((int)threadIdx.x >= offset) add = s[threadIdx.x - offset];
        __syncthreads();
        if ((int)threadIdx.x >= offset) s[threadIdx.x] += add;
        __syncthreads();
    }
    if (threadIdx.x < n) data[threadIdx.x] = s[threadIdx.x];
}
extern "C" __global__ void add_block_offsets(float* out, const float* block_inclusive, size_t n) {
    size_t i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    if (i < n && blockIdx.x > 0) out[i] += block_inclusive[blockIdx.x - 1];
}
extern "C" __global__ void fill_zero(float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}
extern "C" __global__ void histogram(const float* in, float* hist, size_t n, float vmin, float vmax, int bins) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = (in[i] - vmin) / (vmax - vmin);
        if (t < 0.0f) t = 0.0f;
        if (t >= 1.0f) t = 0.999999f;
        int b = (int)(t * (float)bins);
        atomicAdd(&hist[b], 1.0f);
    }
}
"#;

/// Maps a `cudarc` driver error into [`Error::Backend`] with debug formatting.
fn cu_err(err: impl std::fmt::Debug) -> Error {
    Error::Backend(format!("{err:?}"))
}

/// Default 1-D launch configuration: 256 threads per block, enough blocks to cover `n` elements.
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Pool and pinned staging capacity key: smallest power-of-two >= `elems`, at least 256.
fn pool_key(elems: usize) -> usize {
    elems.next_power_of_two().max(256)
}

/// Reusable device allocations indexed by [`pool_key`] size.
///
/// Avoids `cudaMalloc` for ephemeral buffers (prefix block sums, reduction partials) and for
/// resident backing storage returned on `free_device` / shutdown.
struct DevicePool {
    free: HashMap<usize, Vec<CudaSlice<f32>>>,
}

impl DevicePool {
    /// Creates an empty pool (no pre-allocation).
    fn new() -> Self {
        Self {
            free: HashMap::new(),
        }
    }

    /// Returns a device slice with length `pool_key(elems)`; reuses a free list entry or allocates zeros.
    fn acquire(
        &mut self,
        stream: &Arc<CudaStream>,
        elems: usize,
        metrics: &mut RuntimeMetrics,
    ) -> Result<CudaSlice<f32>> {
        let key = pool_key(elems);
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(slice) = list.pop() {
                metrics.pool_hits += 1;
                return Ok(slice);
            }
        }
        metrics.pool_misses += 1;
        stream.alloc_zeros::<f32>(key).map_err(cu_err)
    }

    /// Returns a slice to the free list for its keyed capacity (logical `elems` before rounding).
    fn release(&mut self, elems: usize, slice: CudaSlice<f32>) {
        let key = pool_key(elems);
        self.free.entry(key).or_default().push(slice);
    }
}

/// Device-side mirror of a host buffer: pooled slice plus logical element count.
struct Resident {
    device: CudaSlice<f32>,
    len: usize,
}

/// CUDA events bracketing H2D, kernel, and optional in-stream D2H for one job.
struct StageEvents {
    h2d_start: CudaEvent,
    h2d_end: CudaEvent,
    kernel_end: CudaEvent,
    d2h_end: Option<CudaEvent>,
}

/// Per-stream execution context while a job is in flight or idle.
struct Slot {
    stream: Arc<CudaStream>,
    busy: bool,
    job_id: Option<JobId>,
    done: Option<CudaEvent>,
    buffer_ids: Vec<BufferId>,
    written: Vec<BufferId>,
    /// Reduction partials in pinned host memory; scalar sum applied in `finish_slot` after `done`.
    pending_reduce: Option<(BufferId, PinnedHostSlice<f32>, usize)>,
    temps: Vec<(usize, CudaSlice<f32>)>,
    stages: Option<StageEvents>,
    pinned: Option<PinnedHostSlice<f32>>,
}

/// CUDA implementation of [`Backend`]: NVRTC module, kernel functions, stream slots, pool, residents.
pub(crate) struct CudaBackend {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    /// `y <- alpha * x + y`
    saxpy: CudaFunction,
    /// Element-wise vector add.
    vadd: CudaFunction,
    /// Element-wise scale.
    vscale: CudaFunction,
    /// Block-local sum into partial array (reduction stage 1).
    reduce_block: CudaFunction,
    /// Affine normalize with host-provided mean and inverse std.
    normalize: CudaFunction,
    /// Integrate 6-float particle records (position + velocity) with gravity on Y.
    particle_update: CudaFunction,
    /// Per-256-block inclusive scan + block sum extraction (prefix stage 1).
    scan_blocks: CudaFunction,
    /// Inclusive scan of up to 256 block sums in one block (prefix stage 2).
    scan_small_inclusive: CudaFunction,
    /// Add exclusive prefix of block totals into output (prefix stage 3).
    add_block_offsets: CudaFunction,
    /// Zero-fill histogram output before atomic accumulation.
    fill_zero: CudaFunction,
    /// Bin input values with `atomicAdd` after `fill_zero`.
    histogram: CudaFunction,
    slots: Vec<Slot>,
    device_pool: DevicePool,
    resident: HashMap<u64, Resident>,
}

impl CudaBackend {
    /// Builds context, compiles [`KERNELS`], loads functions, and creates `stream_count` idle slots.
    pub(crate) fn new(stream_count: usize) -> Result<Self> {
        let ctx = CudaContext::new(0).map_err(cu_err)?;
        let ptx = compile_ptx(KERNELS).map_err(cu_err)?;
        let module = ctx.load_module(ptx).map_err(cu_err)?;
        let saxpy = module.load_function("saxpy").map_err(cu_err)?;
        let vadd = module.load_function("vadd").map_err(cu_err)?;
        let vscale = module.load_function("vscale").map_err(cu_err)?;
        let reduce_block = module.load_function("reduce_block").map_err(cu_err)?;
        let normalize = module.load_function("normalize").map_err(cu_err)?;
        let particle_update = module.load_function("particle_update").map_err(cu_err)?;
        let scan_blocks = module.load_function("scan_blocks").map_err(cu_err)?;
        let scan_small_inclusive = module.load_function("scan_small_inclusive").map_err(cu_err)?;
        let add_block_offsets = module.load_function("add_block_offsets").map_err(cu_err)?;
        let fill_zero = module.load_function("fill_zero").map_err(cu_err)?;
        let histogram = module.load_function("histogram").map_err(cu_err)?;

        let mut slots = Vec::with_capacity(stream_count);
        for _ in 0..stream_count {
            let stream = ctx.new_stream().map_err(cu_err)?;
            slots.push(Slot {
                stream,
                busy: false,
                job_id: None,
                done: None,
                buffer_ids: Vec::new(),
                written: Vec::new(),
                pending_reduce: None,
                temps: Vec::new(),
                stages: None,
                pinned: None,
            });
        }

        Ok(Self {
            ctx,
            _module: module,
            saxpy,
            vadd,
            vscale,
            reduce_block,
            normalize,
            particle_update,
            scan_blocks,
            scan_small_inclusive,
            add_block_offsets,
            fill_zero,
            histogram,
            slots,
            device_pool: DevicePool::new(),
            resident: HashMap::new(),
        })
    }

    /// Allocates a CUDA event with default timing flags for stage measurement.
    fn timing_event(&self) -> Result<CudaEvent> {
        self.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(cu_err)
    }

    /// Grows the slot's pinned staging buffer to at least `pool_key(elems)` elements.
    fn ensure_pinned(&mut self, slot_idx: usize, elems: usize) -> Result<()> {
        let need = pool_key(elems);
        let cur = self.slots[slot_idx]
            .pinned
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(0);
        if cur < need {
            // SAFETY: pinned allocation lifetime is managed by `PinnedHostSlice` on drop.
            let pin = unsafe { self.ctx.alloc_pinned::<f32>(need) }.map_err(cu_err)?;
            self.slots[slot_idx].pinned = Some(pin);
        }
        Ok(())
    }

    /// Ensures a resident device allocation exists and is synchronized from host when dirty or new.
    fn ensure_resident(
        &mut self,
        id: BufferId,
        stream: &Arc<CudaStream>,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
        slot_idx: usize,
    ) -> Result<()> {
        let len = memory.len_of(id)?;
        let host_dirty = memory.get(id)?.host_dirty;
        let missing = !self.resident.contains_key(&id.0);

        if missing {
            let device = self.device_pool.acquire(stream, len, metrics)?;
            self.resident.insert(id.0, Resident { device, len });
        }

        if host_dirty || missing {
            self.h2d_buffer(id, stream, memory, metrics, slot_idx)?;
            memory.get_mut(id)?.host_dirty = false;
        }
        Ok(())
    }

    /// Copies host buffer contents through pinned staging into the resident device slice on `stream`.
    fn h2d_buffer(
        &mut self,
        id: BufferId,
        stream: &Arc<CudaStream>,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
        slot_idx: usize,
    ) -> Result<()> {
        let len = memory.len_of(id)?;
        self.ensure_pinned(slot_idx, len)?;

        {
            let host = &memory.get(id)?.host;
            let pin = self.slots[slot_idx].pinned.as_mut().expect("pinned");
            let dst = pin.as_mut_slice().map_err(cu_err)?;
            dst[..len].copy_from_slice(host);
            dst[len..].fill(0.0);
        }

        // `pin.len()` == `device.len()` == `pool_key(len)`; copy the full pooled extent.
        let pin = self.slots[slot_idx].pinned.as_ref().expect("pinned");
        let resident = self
            .resident
            .get_mut(&id.0)
            .ok_or(Error::Backend("resident missing".into()))?;
        if resident.device.len() < pin.len() {
            return Err(Error::Backend("device smaller than pinned staging".into()));
        }
        stream
            .memcpy_htod(pin, &mut resident.device)
            .map_err(cu_err)?;
        metrics.bytes_h2d += (len * 4) as u64;
        Ok(())
    }

    /// Removes and returns the resident entry for kernel launches (caller must `put_resident` after).
    fn take_resident(&mut self, id: BufferId) -> Result<Resident> {
        self.resident
            .remove(&id.0)
            .ok_or(Error::Backend("device buffer missing".into()))
    }

    /// Re-inserts a resident after kernels complete.
    fn put_resident(&mut self, id: BufferId, r: Resident) {
        self.resident.insert(id.0, r);
    }

    /// Tears down slot state after `done`, fills metrics, host reduction, dirty flags, pool release.
    fn finish_slot(
        &mut self,
        idx: usize,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
        failed: Option<String>,
    ) -> CompletedWork {
        let slot = &mut self.slots[idx];
        let job_id = slot.job_id.take().expect("busy");
        let buffer_ids = std::mem::take(&mut slot.buffer_ids);
        let written = std::mem::take(&mut slot.written);
        let pending_reduce = slot.pending_reduce.take();
        let temps = std::mem::take(&mut slot.temps);
        let stages = slot.stages.take();
        slot.done = None;
        slot.busy = false;

        let mut job_metrics = JobMetrics::default();
        if let Some(st) = stages {
            if let Ok(ms) = st.h2d_start.elapsed_ms(&st.h2d_end) {
                job_metrics.h2d_us = (ms as f64 * 1000.0) as u64;
            }
            if let Ok(ms) = st.h2d_end.elapsed_ms(&st.kernel_end) {
                job_metrics.kernel_us = (ms as f64 * 1000.0) as u64;
            }
            if let Some(d2h_end) = st.d2h_end {
                if let Ok(ms) = st.kernel_end.elapsed_ms(&d2h_end) {
                    job_metrics.d2h_us = (ms as f64 * 1000.0) as u64;
                }
                metrics.last_gpu_ms = st.h2d_start.elapsed_ms(&d2h_end).unwrap_or(0.0) as f64;
            } else {
                metrics.last_gpu_ms = st.h2d_start.elapsed_ms(&st.kernel_end).unwrap_or(0.0) as f64;
            }
        }

        if let Some((out_id, pin, npartial)) = pending_reduce {
            if let Ok(slice) = pin.as_slice() {
                let sum: f32 = slice[..npartial].iter().copied().sum();
                if let Ok(entry) = memory.get_mut(out_id) {
                    entry.host[0] = sum;
                    entry.host_dirty = true;
                    entry.device_dirty = false;
                }
                // Device copy of scalar output stays stale until next `ensure_resident` sees `host_dirty`.
            }
        }

        for id in &written {
            if let Ok(entry) = memory.get_mut(*id) {
                if !entry.host_dirty {
                    entry.device_dirty = true;
                }
            }
        }

        for (elems, slice) in temps {
            self.device_pool.release(elems, slice);
        }
        memory.release(&buffer_ids);

        CompletedWork {
            job_id,
            metrics: job_metrics,
            failed,
        }
    }
}

impl Backend for CudaBackend {
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    fn in_flight(&self) -> usize {
        self.slots.iter().filter(|s| s.busy).count()
    }

    /// Records stage events, uploads residents, launches kernels for `op`, and arms `done` on the slot.
    fn try_issue(
        &mut self,
        slot_idx: usize,
        job_id: JobId,
        op: &Operation,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()> {
        if self.slots[slot_idx].busy {
            return Err(Error::Busy);
        }

        let stream = self.slots[slot_idx].stream.clone();
        let buffer_ids = op.buffer_ids();
        let mut temps = Vec::new();
        let mut written = Vec::new();
        let mut pending_reduce = None;

        let h2d_start = self.timing_event()?;
        h2d_start.record(&stream).map_err(cu_err)?;

        for id in op.reads().into_iter().chain(op.writes()) {
            self.ensure_resident(id, &stream, memory, metrics, slot_idx)?;
        }

        let h2d_end = self.timing_event()?;
        h2d_end.record(&stream).map_err(cu_err)?;

        match *op {
            // BLAS-style `y <- alpha * x + y` on device resident slices.
            Operation::Saxpy { alpha, x, y } => {
                let n = memory.len_of(x)?;
                let d_x = self.take_resident(x)?;
                let mut d_y = self.take_resident(y)?;
                let n_arg = n;
                let x_view = d_x.device.slice(..n);
                let mut y_view = d_y.device.slice_mut(..n);
                let mut b = stream.launch_builder(&self.saxpy);
                b.arg(&alpha);
                b.arg(&x_view);
                b.arg(&mut y_view);
                b.arg(&n_arg);
                unsafe { b.launch(launch_cfg(n as u32)) }.map_err(cu_err)?;
                self.put_resident(x, d_x);
                self.put_resident(y, d_y);
                written.push(y);
            }
            // `output[i] = a[i] + b[i]`.
            Operation::VectorAdd { a, b, output } => {
                let n = memory.len_of(a)?;
                let d_a = self.take_resident(a)?;
                let d_b = self.take_resident(b)?;
                let mut d_o = self.take_resident(output)?;
                let n_arg = n;
                let a_view = d_a.device.slice(..n);
                let b_view = d_b.device.slice(..n);
                let mut o_view = d_o.device.slice_mut(..n);
                let mut builder = stream.launch_builder(&self.vadd);
                builder.arg(&a_view);
                builder.arg(&b_view);
                builder.arg(&mut o_view);
                builder.arg(&n_arg);
                unsafe { builder.launch(launch_cfg(n as u32)) }.map_err(cu_err)?;
                self.put_resident(a, d_a);
                self.put_resident(b, d_b);
                self.put_resident(output, d_o);
                written.push(output);
            }
            // `output[i] = alpha * input[i]`.
            Operation::VectorScale {
                alpha,
                input,
                output,
            } => {
                let n = memory.len_of(input)?;
                let d_in = self.take_resident(input)?;
                let mut d_out = self.take_resident(output)?;
                let n_arg = n;
                let in_view = d_in.device.slice(..n);
                let mut out_view = d_out.device.slice_mut(..n);
                let mut builder = stream.launch_builder(&self.vscale);
                builder.arg(&alpha);
                builder.arg(&in_view);
                builder.arg(&mut out_view);
                builder.arg(&n_arg);
                unsafe { builder.launch(launch_cfg(n as u32)) }.map_err(cu_err)?;
                self.put_resident(input, d_in);
                self.put_resident(output, d_out);
                written.push(output);
            }
            // Block partial sums D2H to pinned buffer; final add on host in `finish_slot`.
            Operation::ReductionSum { input, output } => {
                let n = memory.len_of(input)?;
                let blocks = ((n as u32) + 255) / 256;
                let d_in = self.take_resident(input)?;
                let mut d_partial = self
                    .device_pool
                    .acquire(&stream, blocks as usize, metrics)?;
                let n_arg = n;
                let in_view = d_in.device.slice(..n);
                let mut builder = stream.launch_builder(&self.reduce_block);
                builder.arg(&in_view);
                builder.arg(&mut d_partial);
                builder.arg(&n_arg);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(cu_err)?;
                self.put_resident(input, d_in);

                // Dedicated pinned buffer for partials (does not reuse slot H2D staging).
                let mut pin =
                    unsafe { self.ctx.alloc_pinned::<f32>(blocks as usize) }.map_err(cu_err)?;
                let view = d_partial.slice(..blocks as usize);
                stream.memcpy_dtoh(&view, &mut pin).map_err(cu_err)?;
                metrics.bytes_d2h += (blocks as u64) * 4;
                pending_reduce = Some((output, pin, blocks as usize));
                temps.push((blocks as usize, d_partial));
                written.push(output);
            }
            // `(input - mean) * inv_std` element-wise.
            Operation::Normalize {
                input,
                output,
                mean,
                inv_std,
            } => {
                let n = memory.len_of(input)?;
                let d_in = self.take_resident(input)?;
                let mut d_out = self.take_resident(output)?;
                let n_arg = n;
                let in_view = d_in.device.slice(..n);
                let mut out_view = d_out.device.slice_mut(..n);
                let mut builder = stream.launch_builder(&self.normalize);
                builder.arg(&in_view);
                builder.arg(&mut out_view);
                builder.arg(&mean);
                builder.arg(&inv_std);
                builder.arg(&n_arg);
                unsafe { builder.launch(launch_cfg(n as u32)) }.map_err(cu_err)?;
                self.put_resident(input, d_in);
                self.put_resident(output, d_out);
                written.push(output);
            }
            // Six floats per particle: integrate velocity with gravity, then positions.
            Operation::ParticleUpdate {
                state,
                delta_time,
                gravity_y,
            } => {
                let n_floats = memory.len_of(state)?;
                let n_particles = n_floats / 6;
                let mut d_state = self.take_resident(state)?;
                let n_arg = n_particles;
                let mut view = d_state.device.slice_mut(..n_floats);
                let mut builder = stream.launch_builder(&self.particle_update);
                builder.arg(&mut view);
                builder.arg(&delta_time);
                builder.arg(&gravity_y);
                builder.arg(&n_arg);
                unsafe { builder.launch(launch_cfg(n_particles as u32)) }.map_err(cu_err)?;
                self.put_resident(state, d_state);
                written.push(state);
            }
            // Two-level inclusive prefix sum: `scan_blocks`, `scan_small_inclusive`, `add_block_offsets`.
            Operation::PrefixSum { input, output } => {
                const SCAN_BLOCK: u32 = 256;
                let n = memory.len_of(input)?;
                let blocks = ((n as u32) + SCAN_BLOCK - 1) / SCAN_BLOCK;
                if blocks > 256 {
                    return Err(Error::Invalid(
                        "prefix_sum CUDA: n > 65536 not supported in 0.1",
                    ));
                }
                let d_in = self.take_resident(input)?;
                let mut d_out = self.take_resident(output)?;
                let mut d_block = self
                    .device_pool
                    .acquire(&stream, blocks as usize, metrics)?;
                let n_arg = n;
                let in_view = d_in.device.slice(..n);
                let mut out_view = d_out.device.slice_mut(..n);
                let mut builder = stream.launch_builder(&self.scan_blocks);
                builder.arg(&in_view);
                builder.arg(&mut out_view);
                builder.arg(&mut d_block);
                builder.arg(&n_arg);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (SCAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(cu_err)?;

                let blocks_usize = blocks as usize;
                let mut block_view = d_block.slice_mut(..blocks_usize);
                let n_blocks = blocks_usize;
                let mut builder = stream.launch_builder(&self.scan_small_inclusive);
                builder.arg(&mut block_view);
                builder.arg(&n_blocks);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (SCAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(cu_err)?;

                let block_ro = d_block.slice(..blocks_usize);
                let mut out_view = d_out.device.slice_mut(..n);
                let mut builder = stream.launch_builder(&self.add_block_offsets);
                builder.arg(&mut out_view);
                builder.arg(&block_ro);
                builder.arg(&n_arg);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (SCAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(cu_err)?;

                self.put_resident(input, d_in);
                self.put_resident(output, d_out);
                temps.push((blocks as usize, d_block));
                written.push(output);
            }
            // Clear bins with `fill_zero`, then `histogram` with `atomicAdd` per sample.
            Operation::Histogram {
                input,
                output,
                min_v,
                max_v,
            } => {
                let n = memory.len_of(input)?;
                let bins = memory.len_of(output)?;
                let d_in = self.take_resident(input)?;
                let mut d_out = self.take_resident(output)?;
                let n_bins = bins;
                let mut out_view = d_out.device.slice_mut(..bins);
                let mut builder = stream.launch_builder(&self.fill_zero);
                builder.arg(&mut out_view);
                builder.arg(&n_bins);
                unsafe { builder.launch(launch_cfg(bins as u32)) }.map_err(cu_err)?;

                let n_arg = n;
                let bins_i = bins as i32;
                let in_view = d_in.device.slice(..n);
                let mut out_view = d_out.device.slice_mut(..bins);
                let mut builder = stream.launch_builder(&self.histogram);
                builder.arg(&in_view);
                builder.arg(&mut out_view);
                builder.arg(&n_arg);
                builder.arg(&min_v);
                builder.arg(&max_v);
                builder.arg(&bins_i);
                unsafe { builder.launch(launch_cfg(n as u32)) }.map_err(cu_err)?;

                self.put_resident(input, d_in);
                self.put_resident(output, d_out);
                written.push(output);
            }
        }

        let kernel_end = self.timing_event()?;
        kernel_end.record(&stream).map_err(cu_err)?;

        let d2h_end = if pending_reduce.is_some() {
            let ev = self.timing_event()?;
            ev.record(&stream).map_err(cu_err)?;
            Some(ev)
        } else {
            None
        };

        let done = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map_err(cu_err)?;
        done.record(&stream).map_err(cu_err)?;

        let slot = &mut self.slots[slot_idx];
        slot.busy = true;
        slot.job_id = Some(job_id);
        slot.done = Some(done);
        slot.buffer_ids = buffer_ids;
        slot.written = written;
        slot.pending_reduce = pending_reduce;
        slot.temps = temps;
        slot.stages = Some(StageEvents {
            h2d_start,
            h2d_end,
            kernel_end,
            d2h_end,
        });
        Ok(())
    }

    /// Non-blocking poll of each slot's `done` event; completes slots via `finish_slot`.
    fn poll_completions(
        &mut self,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<Vec<CompletedWork>> {
        let mut out = Vec::new();
        let n = self.slots.len();
        for i in 0..n {
            if !self.slots[i].busy {
                continue;
            }
            let event = self.slots[i].done.as_ref().expect("busy without event");
            match event.try_is_complete() {
                Ok(true) => out.push(self.finish_slot(i, memory, metrics, None)),
                Ok(false) => {}
                Err(e) => {
                    out.push(self.finish_slot(i, memory, metrics, Some(format!("{e:?}"))));
                }
            }
        }
        Ok(out)
    }

    /// Synchronous D2H on stream 0 when `device_dirty`; copies pinned slice into host memory.
    fn prepare_download(
        &mut self,
        id: BufferId,
        memory: &mut MemoryManager,
        metrics: &mut RuntimeMetrics,
    ) -> Result<()> {
        let entry = memory.get(id)?;
        if !entry.device_dirty {
            return Ok(());
        }
        let len = entry.len;
        if !self.resident.contains_key(&id.0) {
            return Err(Error::Backend("download: device missing".into()));
        }

        // Stream 0 performs synchronous download D2H.
        let stream = self.slots[0].stream.clone();
        self.ensure_pinned(0, len)?;

        {
            let resident = self.resident.get(&id.0).expect("resident");
            let view = resident.device.slice(..len);
            let pin = self.slots[0].pinned.as_mut().expect("pinned");
            stream.memcpy_dtoh(&view, pin).map_err(cu_err)?;
            stream.synchronize().map_err(cu_err)?;
            metrics.bytes_d2h += (len * 4) as u64;
        }

        {
            let pin = self.slots[0].pinned.as_ref().expect("pinned");
            let src = pin.as_slice().map_err(cu_err)?;
            let entry = memory.get_mut(id)?;
            entry.host.copy_from_slice(&src[..len]);
            entry.device_dirty = false;
            entry.host_dirty = false;
        }
        Ok(())
    }

    /// Drops resident mapping and returns device memory to the pool.
    fn free_device(&mut self, id: BufferId, metrics: &mut RuntimeMetrics) -> Result<()> {
        if let Some(r) = self.resident.remove(&id.0) {
            self.device_pool.release(r.len, r.device);
            let _ = metrics;
        }
        Ok(())
    }

    /// Waits on in-flight events and returns all resident allocations to the pool.
    fn shutdown(&mut self) -> Result<()> {
        for slot in &mut self.slots {
            if let Some(ev) = slot.done.take() {
                let _ = ev.synchronize();
            }
            slot.busy = false;
            slot.job_id = None;
            slot.pending_reduce = None;
            slot.temps.clear();
            slot.buffer_ids.clear();
            slot.written.clear();
            slot.stages = None;
        }
        // Return all residents to the device pool.
        let keys: Vec<u64> = self.resident.keys().copied().collect();
        for k in keys {
            if let Some(r) = self.resident.remove(&k) {
                self.device_pool.release(r.len, r.device);
            }
        }
        Ok(())
    }
}
