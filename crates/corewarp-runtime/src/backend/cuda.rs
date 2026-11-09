//! CUDA: buffers residentes, pinned staging e timing por CudaEvent.

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

fn cu_err(err: impl std::fmt::Debug) -> Error {
    Error::Backend(format!("{err:?}"))
}

fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pool_key(elems: usize) -> usize {
    elems.next_power_of_two().max(256)
}

struct DevicePool {
    free: HashMap<usize, Vec<CudaSlice<f32>>>,
}

impl DevicePool {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
        }
    }

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

    fn release(&mut self, elems: usize, slice: CudaSlice<f32>) {
        let key = pool_key(elems);
        self.free.entry(key).or_default().push(slice);
    }
}

struct Resident {
    device: CudaSlice<f32>,
    len: usize,
}

struct StageEvents {
    h2d_start: CudaEvent,
    h2d_end: CudaEvent,
    kernel_end: CudaEvent,
    d2h_end: Option<CudaEvent>,
}

struct Slot {
    stream: Arc<CudaStream>,
    busy: bool,
    job_id: Option<JobId>,
    done: Option<CudaEvent>,
    buffer_ids: Vec<BufferId>,
    written: Vec<BufferId>,
    /// Partials pinned para reduction (somatório só após done).
    pending_reduce: Option<(BufferId, PinnedHostSlice<f32>, usize)>,
    temps: Vec<(usize, CudaSlice<f32>)>,
    stages: Option<StageEvents>,
    pinned: Option<PinnedHostSlice<f32>>,
}

pub(crate) struct CudaBackend {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    saxpy: CudaFunction,
    vadd: CudaFunction,
    vscale: CudaFunction,
    reduce_block: CudaFunction,
    normalize: CudaFunction,
    particle_update: CudaFunction,
    scan_blocks: CudaFunction,
    scan_small_inclusive: CudaFunction,
    add_block_offsets: CudaFunction,
    fill_zero: CudaFunction,
    histogram: CudaFunction,
    slots: Vec<Slot>,
    device_pool: DevicePool,
    resident: HashMap<u64, Resident>,
}

impl CudaBackend {
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

    fn timing_event(&self) -> Result<CudaEvent> {
        self.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(cu_err)
    }

    fn ensure_pinned(&mut self, slot_idx: usize, elems: usize) -> Result<()> {
        let need = pool_key(elems);
        let cur = self.slots[slot_idx]
            .pinned
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(0);
        if cur < need {
            // SAFETY: alocação pinned gerenciada pelo Drop de PinnedHostSlice.
            let pin = unsafe { self.ctx.alloc_pinned::<f32>(need) }.map_err(cu_err)?;
            self.slots[slot_idx].pinned = Some(pin);
        }
        Ok(())
    }

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

        // pin.len() == device.len() == pool_key(len); copia bloco inteiro.
        let pin = self.slots[slot_idx].pinned.as_ref().expect("pinned");
        let resident = self
            .resident
            .get_mut(&id.0)
            .ok_or(Error::Backend("resident ausente".into()))?;
        if resident.device.len() < pin.len() {
            return Err(Error::Backend("device menor que pinned staging".into()));
        }
        stream
            .memcpy_htod(pin, &mut resident.device)
            .map_err(cu_err)?;
        metrics.bytes_h2d += (len * 4) as u64;
        Ok(())
    }

    fn take_resident(&mut self, id: BufferId) -> Result<Resident> {
        self.resident
            .remove(&id.0)
            .ok_or(Error::Backend("buffer device ausente".into()))
    }

    fn put_resident(&mut self, id: BufferId, r: Resident) {
        self.resident.insert(id.0, r);
    }

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
                // Mantém device output coerente no próximo ensure (host_dirty).
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

                // Pinned dedicado aos parciais (não reutiliza staging H2D do slot).
                let mut pin =
                    unsafe { self.ctx.alloc_pinned::<f32>(blocks as usize) }.map_err(cu_err)?;
                let view = d_partial.slice(..blocks as usize);
                stream.memcpy_dtoh(&view, &mut pin).map_err(cu_err)?;
                metrics.bytes_d2h += (blocks as u64) * 4;
                pending_reduce = Some((output, pin, blocks as usize));
                temps.push((blocks as usize, d_partial));
                written.push(output);
            }
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
            Operation::PrefixSum { input, output } => {
                const SCAN_BLOCK: u32 = 256;
                let n = memory.len_of(input)?;
                let blocks = ((n as u32) + SCAN_BLOCK - 1) / SCAN_BLOCK;
                if blocks > 256 {
                    return Err(Error::Invalid(
                        "prefix_sum CUDA: n > 65536 não suportado no 0.1",
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
            let event = self.slots[i].done.as_ref().expect("busy sem event");
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
            return Err(Error::Backend("download: device ausente".into()));
        }

        // Usa stream 0 para D2H síncrono de download.
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

    fn free_device(&mut self, id: BufferId, metrics: &mut RuntimeMetrics) -> Result<()> {
        if let Some(r) = self.resident.remove(&id.0) {
            self.device_pool.release(r.len, r.device);
            let _ = metrics;
        }
        Ok(())
    }

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
        // Devolve residentes ao pool.
        let keys: Vec<u64> = self.resident.keys().copied().collect();
        for k in keys {
            if let Some(r) = self.resident.remove(&k) {
                self.device_pool.release(r.len, r.device);
            }
        }
        Ok(())
    }
}
