use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobHandle {
    pub id: JobId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Submitted,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    Fixed,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Host,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub stream_count: usize,
    pub max_in_flight: usize,
    pub issue_width: usize,
    pub memory_pool_bytes: usize,
    pub scheduling: SchedulingPolicy,
    pub backend: BackendKind,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            stream_count: 4,
            max_in_flight: 32,
            issue_width: 2,
            memory_pool_bytes: 128 * 1024 * 1024,
            scheduling: SchedulingPolicy::Fixed,
            backend: BackendKind::Cuda,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub backend: &'static str,
    pub index: u32,
    pub name: String,
}

/// Operações tipadas sobre handles de buffer.
#[derive(Debug, Clone)]
pub enum Operation {
    Saxpy {
        alpha: f32,
        x: BufferId,
        y: BufferId,
    },
    VectorAdd {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },
    VectorScale {
        alpha: f32,
        input: BufferId,
        output: BufferId,
    },
    ReductionSum {
        input: BufferId,
        output: BufferId,
    },
    /// `out[i] = (in[i] - mean) * inv_std`
    Normalize {
        input: BufferId,
        output: BufferId,
        mean: f32,
        inv_std: f32,
    },
    /// Estado AoS: `[px,py,pz,vx,vy,vz] * N`. Atualização Euler in-place.
    ParticleUpdate {
        state: BufferId,
        delta_time: f32,
        gravity_y: f32,
    },
    /// Scan inclusivo: `out[i] = sum(in[0..=i])`. Mesmo comprimento.
    PrefixSum {
        input: BufferId,
        output: BufferId,
    },
    /// Histograma: `output.len()` = número de bins em `[min_v, max_v]`.
    Histogram {
        input: BufferId,
        output: BufferId,
        min_v: f32,
        max_v: f32,
    },
}

impl Operation {
    pub fn buffer_ids(&self) -> Vec<BufferId> {
        let mut ids = self.reads();
        for w in self.writes() {
            if !ids.contains(&w) {
                ids.push(w);
            }
        }
        ids
    }

    pub fn reads(&self) -> Vec<BufferId> {
        match *self {
            Self::Saxpy { x, y, .. } => {
                if x == y {
                    vec![x]
                } else {
                    vec![x, y]
                }
            }
            Self::VectorAdd { a, b, .. } => vec![a, b],
            Self::VectorScale { input, .. } => vec![input],
            Self::ReductionSum { input, .. } => vec![input],
            Self::Normalize { input, .. } => vec![input],
            Self::ParticleUpdate { state, .. } => vec![state],
            Self::PrefixSum { input, .. } => vec![input],
            Self::Histogram { input, .. } => vec![input],
        }
    }

    pub fn writes(&self) -> Vec<BufferId> {
        match *self {
            Self::Saxpy { y, .. } => vec![y],
            Self::VectorAdd { output, .. } => vec![output],
            Self::VectorScale { output, .. } => vec![output],
            Self::ReductionSum { output, .. } => vec![output],
            Self::Normalize { output, .. } => vec![output],
            Self::ParticleUpdate { state, .. } => vec![state],
            Self::PrefixSum { output, .. } => vec![output],
            Self::Histogram { output, .. } => vec![output],
        }
    }

    pub fn conflicts_with(&self, active_reads: &[BufferId], active_writes: &[BufferId]) -> bool {
        for w in self.writes() {
            if active_writes.contains(&w) || active_reads.contains(&w) {
                return true;
            }
        }
        for r in self.reads() {
            if active_writes.contains(&r) {
                return true;
            }
        }
        false
    }

    pub fn validate_shapes(&self, len_of: impl Fn(BufferId) -> Result<usize>) -> Result<()> {
        match *self {
            Self::Saxpy { x, y, .. } => {
                let nx = len_of(x)?;
                let ny = len_of(y)?;
                if nx == 0 || nx != ny {
                    return Err(Error::Invalid("saxpy: x/y incompatíveis"));
                }
            }
            Self::VectorAdd { a, b, output } => {
                let na = len_of(a)?;
                let nb = len_of(b)?;
                let no = len_of(output)?;
                if na == 0 || na != nb || na != no {
                    return Err(Error::Invalid("vector_add: tamanhos incompatíveis"));
                }
            }
            Self::VectorScale { input, output, .. } | Self::Normalize { input, output, .. } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 || ni != no {
                    return Err(Error::Invalid("scale/normalize: tamanhos incompatíveis"));
                }
            }
            Self::ReductionSum { input, output } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 {
                    return Err(Error::Invalid("reduction: input vazio"));
                }
                if no != 1 {
                    return Err(Error::Invalid("reduction: output deve ter len=1"));
                }
            }
            Self::ParticleUpdate { state, .. } => {
                let n = len_of(state)?;
                if n == 0 || n % 6 != 0 {
                    return Err(Error::Invalid("particle: len deve ser múltiplo de 6"));
                }
            }
            Self::PrefixSum { input, output } => {
                let ni = len_of(input)?;
                let no = len_of(output)?;
                if ni == 0 || ni != no {
                    return Err(Error::Invalid("prefix_sum: x/y incompatíveis"));
                }
            }
            Self::Histogram {
                input,
                output,
                min_v,
                max_v,
            } => {
                let ni = len_of(input)?;
                let bins = len_of(output)?;
                if ni == 0 || bins == 0 {
                    return Err(Error::Invalid("histogram: input/bins vazios"));
                }
                if !(max_v > min_v) {
                    return Err(Error::Invalid("histogram: max_v deve ser > min_v"));
                }
            }
        }
        Ok(())
    }
}
