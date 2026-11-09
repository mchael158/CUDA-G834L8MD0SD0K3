//! Pool host por size-class + dirty flags para coerência host/device.

use std::collections::HashMap;

use corewarp_core::BufferId;
use corewarp_core::{Error, Result};
use corewarp_telemetry::RuntimeMetrics;

const CLASS_SMALL: usize = 4 * 1024;
const CLASS_MEDIUM: usize = 64 * 1024;
const CLASS_LARGE: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    Small,
    Medium,
    Large,
    Extra,
}

impl SizeClass {
    pub fn for_bytes(bytes: usize) -> Self {
        if bytes <= CLASS_SMALL {
            Self::Small
        } else if bytes <= CLASS_MEDIUM {
            Self::Medium
        } else if bytes <= CLASS_LARGE {
            Self::Large
        } else {
            Self::Extra
        }
    }

    pub fn capacity_bytes(self) -> Option<usize> {
        match self {
            Self::Small => Some(CLASS_SMALL),
            Self::Medium => Some(CLASS_MEDIUM),
            Self::Large => Some(CLASS_LARGE),
            Self::Extra => None,
        }
    }
}

#[derive(Debug)]
pub struct BufferEntry {
    pub len: usize,
    pub host: Vec<f32>,
    pub in_flight: u32,
    pub class: SizeClass,
    /// Host tem dados mais novos que o device.
    pub host_dirty: bool,
    /// Device tem dados mais novos que o host.
    pub device_dirty: bool,
}

struct FreeBlock {
    capacity_elems: usize,
    storage: Vec<f32>,
}

pub struct MemoryManager {
    next_id: u64,
    buffers: HashMap<u64, BufferEntry>,
    free_small: Vec<FreeBlock>,
    free_medium: Vec<FreeBlock>,
    free_large: Vec<FreeBlock>,
    budget_bytes: usize,
    used_bytes: usize,
}

impl MemoryManager {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            next_id: 1,
            buffers: HashMap::new(),
            free_small: Vec::new(),
            free_medium: Vec::new(),
            free_large: Vec::new(),
            budget_bytes,
            used_bytes: 0,
        }
    }

    pub fn get(&self, id: BufferId) -> Result<&BufferEntry> {
        self.buffers
            .get(&id.0)
            .ok_or(Error::Invalid("buffer inexistente"))
    }

    pub fn get_mut(&mut self, id: BufferId) -> Result<&mut BufferEntry> {
        self.buffers
            .get_mut(&id.0)
            .ok_or(Error::Invalid("buffer inexistente"))
    }

    pub fn len_of(&self, id: BufferId) -> Result<usize> {
        Ok(self.get(id)?.len)
    }

    pub fn upload(&mut self, data: &[f32], metrics: &mut RuntimeMetrics) -> Result<BufferId> {
        if data.is_empty() {
            return Err(Error::Invalid("upload: slice vazio"));
        }
        let bytes = std::mem::size_of_val(data);
        let class = SizeClass::for_bytes(bytes);
        let mut storage = self.acquire(class, data.len(), metrics)?;
        storage[..data.len()].copy_from_slice(data);
        storage.truncate(data.len());

        let id = BufferId(self.next_id);
        self.next_id += 1;
        self.buffers.insert(
            id.0,
            BufferEntry {
                len: data.len(),
                host: storage,
                in_flight: 0,
                class,
                host_dirty: true,
                device_dirty: false,
            },
        );
        Ok(id)
    }

    pub fn alloc_zeros(&mut self, len: usize, metrics: &mut RuntimeMetrics) -> Result<BufferId> {
        if len == 0 {
            return Err(Error::Invalid("alloc: len=0"));
        }
        let bytes = len * std::mem::size_of::<f32>();
        let class = SizeClass::for_bytes(bytes);
        let mut storage = self.acquire(class, len, metrics)?;
        storage[..len].fill(0.0);
        storage.truncate(len);

        let id = BufferId(self.next_id);
        self.next_id += 1;
        self.buffers.insert(
            id.0,
            BufferEntry {
                len,
                host: storage,
                in_flight: 0,
                class,
                host_dirty: true,
                device_dirty: false,
            },
        );
        Ok(id)
    }

    pub fn download(&self, id: BufferId) -> Result<Vec<f32>> {
        let entry = self.get(id)?;
        if entry.in_flight > 0 {
            return Err(Error::NotReady);
        }
        if entry.device_dirty {
            return Err(Error::NotReady);
        }
        Ok(entry.host.clone())
    }

    pub fn retain(&mut self, ids: &[BufferId]) -> Result<()> {
        for id in ids {
            self.get_mut(*id)?.in_flight += 1;
        }
        Ok(())
    }

    pub fn release(&mut self, ids: &[BufferId]) {
        for id in ids {
            if let Ok(entry) = self.get_mut(*id) {
                entry.in_flight = entry.in_flight.saturating_sub(1);
            }
        }
    }

    pub fn free(&mut self, id: BufferId) -> Result<()> {
        let entry = self
            .buffers
            .remove(&id.0)
            .ok_or(Error::Invalid("buffer inexistente"))?;
        if entry.in_flight > 0 {
            self.buffers.insert(id.0, entry);
            return Err(Error::Busy);
        }
        self.recycle(entry.class, entry.host);
        Ok(())
    }

    fn acquire(
        &mut self,
        class: SizeClass,
        need_elems: usize,
        metrics: &mut RuntimeMetrics,
    ) -> Result<Vec<f32>> {
        let list = match class {
            SizeClass::Small => &mut self.free_small,
            SizeClass::Medium => &mut self.free_medium,
            SizeClass::Large => &mut self.free_large,
            SizeClass::Extra => {
                metrics.pool_misses += 1;
                let bytes = need_elems * std::mem::size_of::<f32>();
                self.charge(bytes)?;
                return Ok(vec![0.0; need_elems]);
            }
        };

        if let Some(idx) = list.iter().position(|b| b.capacity_elems >= need_elems) {
            metrics.pool_hits += 1;
            let mut block = list.swap_remove(idx);
            if block.storage.len() < need_elems {
                block.storage.resize(need_elems, 0.0);
            }
            return Ok(block.storage);
        }

        metrics.pool_misses += 1;
        let cap = class.capacity_bytes().unwrap_or(need_elems * 4) / std::mem::size_of::<f32>();
        let elems = cap.max(need_elems);
        let bytes = elems * std::mem::size_of::<f32>();
        self.charge(bytes)?;
        Ok(vec![0.0; elems])
    }

    fn recycle(&mut self, class: SizeClass, storage: Vec<f32>) {
        let capacity_elems = storage.capacity().max(storage.len());
        let block = FreeBlock {
            capacity_elems,
            storage,
        };
        match class {
            SizeClass::Small => self.free_small.push(block),
            SizeClass::Medium => self.free_medium.push(block),
            SizeClass::Large => self.free_large.push(block),
            SizeClass::Extra => {
                let bytes = block.storage.capacity() * std::mem::size_of::<f32>();
                self.used_bytes = self.used_bytes.saturating_sub(bytes);
            }
        }
    }

    fn charge(&mut self, bytes: usize) -> Result<()> {
        if self.used_bytes + bytes > self.budget_bytes {
            return Err(Error::Oom);
        }
        self.used_bytes += bytes;
        Ok(())
    }
}
