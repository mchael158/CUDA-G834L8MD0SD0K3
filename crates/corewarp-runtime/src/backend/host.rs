//! Backend host: valida contratos e produz resultados corretos.
//! Tempos aqui NÃO representam comportamento de GPU.

use crate::backend::{Backend, CompletedWork};
use corewarp_core::{BufferId, JobId, Operation};
use corewarp_core::{Error, Result};
use corewarp_memory::MemoryManager;
use corewarp_telemetry::{JobMetrics, RuntimeMetrics};

struct Slot {
    busy: bool,
    job_id: Option<JobId>,
    buffer_ids: Vec<BufferId>,
}

pub(crate) struct HostBackend {
    slots: Vec<Slot>,
}

impl HostBackend {
    pub(crate) fn new(stream_count: usize) -> Self {
        let slots = (0..stream_count)
            .map(|_| Slot {
                busy: false,
                job_id: None,
                buffer_ids: Vec::new(),
            })
            .collect();
        Self { slots }
    }

    fn run_op(op: &Operation, memory: &mut MemoryManager) -> Result<()> {
        match *op {
            Operation::Saxpy { alpha, x, y } => {
                let nx = memory.len_of(x)?;
                let x_data = memory.get(x)?.host.clone();
                let y_entry = memory.get_mut(y)?;
                for i in 0..nx {
                    y_entry.host[i] += alpha * x_data[i];
                }
                y_entry.host_dirty = false;
                y_entry.device_dirty = false;
            }
            Operation::VectorAdd { a, b, output } => {
                let n = memory.len_of(a)?;
                let a_data = memory.get(a)?.host.clone();
                let b_data = memory.get(b)?.host.clone();
                let out = memory.get_mut(output)?;
                for i in 0..n {
                    out.host[i] = a_data[i] + b_data[i];
                }
                out.host_dirty = false;
                out.device_dirty = false;
            }
            Operation::VectorScale {
                alpha,
                input,
                output,
            } => {
                let n = memory.len_of(input)?;
                let input_data = memory.get(input)?.host.clone();
                let out = memory.get_mut(output)?;
                for i in 0..n {
                    out.host[i] = alpha * input_data[i];
                }
                out.host_dirty = false;
                out.device_dirty = false;
            }
            Operation::ReductionSum { input, output } => {
                let sum: f32 = memory.get(input)?.host.iter().sum();
                let out = memory.get_mut(output)?;
                out.host[0] = sum;
                out.host_dirty = false;
                out.device_dirty = false;
            }
            Operation::Normalize {
                input,
                output,
                mean,
                inv_std,
            } => {
                let n = memory.len_of(input)?;
                let input_data = memory.get(input)?.host.clone();
                let out = memory.get_mut(output)?;
                for i in 0..n {
                    out.host[i] = (input_data[i] - mean) * inv_std;
                }
                out.host_dirty = false;
                out.device_dirty = false;
            }
            Operation::ParticleUpdate {
                state,
                delta_time,
                gravity_y,
            } => {
                let s = memory.get_mut(state)?;
                let n = s.host.len() / 6;
                for i in 0..n {
                    let base = i * 6;
                    s.host[base + 4] += gravity_y * delta_time; // vy
                    s.host[base] += s.host[base + 3] * delta_time;
                    s.host[base + 1] += s.host[base + 4] * delta_time;
                    s.host[base + 2] += s.host[base + 5] * delta_time;
                }
                s.host_dirty = false;
                s.device_dirty = false;
            }
            Operation::PrefixSum { input, output } => {
                let data = memory.get(input)?.host.clone();
                let out = memory.get_mut(output)?;
                let mut acc = 0.0f32;
                for (i, v) in data.iter().enumerate() {
                    acc += *v;
                    out.host[i] = acc;
                }
                out.host_dirty = false;
                out.device_dirty = false;
            }
            Operation::Histogram {
                input,
                output,
                min_v,
                max_v,
            } => {
                let data = memory.get(input)?.host.clone();
                let bins = memory.len_of(output)?;
                let out = memory.get_mut(output)?;
                out.host.fill(0.0);
                let span = max_v - min_v;
                for v in data {
                    let mut t = (v - min_v) / span;
                    if t < 0.0 {
                        t = 0.0;
                    }
                    if t >= 1.0 {
                        t = 0.999_999;
                    }
                    let b = (t * bins as f32) as usize;
                    out.host[b] += 1.0;
                }
                out.host_dirty = false;
                out.device_dirty = false;
            }
        }
        Ok(())
    }
}

impl Backend for HostBackend {
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    fn in_flight(&self) -> usize {
        self.slots.iter().filter(|s| s.busy).count()
    }

    fn try_issue(
        &mut self,
        slot: usize,
        job_id: JobId,
        op: &Operation,
        memory: &mut MemoryManager,
        _metrics: &mut RuntimeMetrics,
    ) -> Result<()> {
        if self.slots[slot].busy {
            return Err(Error::Busy);
        }
        let ids = op.buffer_ids();
        Self::run_op(op, memory)?;
        self.slots[slot].busy = true;
        self.slots[slot].job_id = Some(job_id);
        self.slots[slot].buffer_ids = ids;
        Ok(())
    }

    fn poll_completions(
        &mut self,
        memory: &mut MemoryManager,
        _metrics: &mut RuntimeMetrics,
    ) -> Result<Vec<CompletedWork>> {
        let mut out = Vec::new();
        for slot in &mut self.slots {
            if !slot.busy {
                continue;
            }
            let job_id = slot.job_id.take().expect("busy");
            memory.release(&slot.buffer_ids);
            slot.buffer_ids.clear();
            slot.busy = false;
            out.push(CompletedWork {
                job_id,
                metrics: JobMetrics::default(),
                failed: None,
            });
        }
        Ok(out)
    }

    fn prepare_download(
        &mut self,
        _id: BufferId,
        _memory: &mut MemoryManager,
        _metrics: &mut RuntimeMetrics,
    ) -> Result<()> {
        Ok(())
    }

    fn free_device(&mut self, _id: BufferId, _metrics: &mut RuntimeMetrics) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        for slot in &mut self.slots {
            slot.busy = false;
            slot.job_id = None;
            slot.buffer_ids.clear();
        }
        Ok(())
    }
}
