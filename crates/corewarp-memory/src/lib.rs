//! # corewarp-memory
//!
//! Host-side size-class memory pool plus coherence dirty flags for host↔device data.
//!
//! ## Responsibilities
//! 1. Allocate / recycle `Vec<f32>` storage for buffers under a byte budget.
//! 2. Track per-buffer metadata: element length, size class, in-flight refcount,
//!    and dirty bits that tell the CUDA backend when H2D / D2H is required.
//! 3. Refuse downloads while work is in-flight or while the device holds newer data
//!    (`device_dirty`) until the backend syncs via `prepare_download`.
//!
//! ## Size classes
//! Fixed buckets reduce allocator churn for common tensor sizes:
//! - Small  ≤ 4 KiB
//! - Medium ≤ 64 KiB
//! - Large  ≤ 1 MiB
//! - Extra  > 1 MiB (no freelist; charged/released directly against the budget)
//!
//! ## Dirty-flag protocol
//! - `host_dirty = true`  → host storage is authoritative; CUDA must H2D before use.
//! - `device_dirty = true` → device storage is authoritative; download must D2H first.
//! - After a host-backend op both flags are cleared (host is already coherent).
//! - After a CUDA write, `device_dirty` is set unless the host was freshly written.

use std::collections::HashMap;

use corewarp_core::BufferId;
use corewarp_core::{Error, Result};
use corewarp_telemetry::RuntimeMetrics;

/// Upper bound (bytes) for the Small freelist class.
const CLASS_SMALL: usize = 4 * 1024;
/// Upper bound (bytes) for the Medium freelist class.
const CLASS_MEDIUM: usize = 64 * 1024;
/// Upper bound (bytes) for the Large freelist class.
const CLASS_LARGE: usize = 1024 * 1024;

/// Discrete allocation bucket used by the host pool freelists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    Small,
    Medium,
    Large,
    /// Oversized allocations skip freelist reuse and free their charge on recycle.
    Extra,
}

impl SizeClass {
    /// Map a requested byte size onto the smallest class that can hold it.
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

    /// Fixed capacity in bytes for freelist classes; `None` for Extra.
    pub fn capacity_bytes(self) -> Option<usize> {
        match self {
            Self::Small => Some(CLASS_SMALL),
            Self::Medium => Some(CLASS_MEDIUM),
            Self::Large => Some(CLASS_LARGE),
            Self::Extra => None,
        }
    }
}

/// Live buffer entry owned by [`MemoryManager`].
#[derive(Debug)]
pub struct BufferEntry {
    /// Logical element count visible to ops (`host` may have larger capacity).
    pub len: usize,
    /// Host-visible `f32` storage (authoritative when `!device_dirty`).
    pub host: Vec<f32>,
    /// Number of queued/running jobs that still reference this buffer.
    /// Download and free refuse while this is non-zero.
    pub in_flight: u32,
    pub class: SizeClass,
    /// Host has newer data than the device resident copy.
    pub host_dirty: bool,
    /// Device has newer data than the host mirror.
    pub device_dirty: bool,
}

/// Recycled storage block sitting on a size-class freelist.
struct FreeBlock {
    /// Element capacity previously reserved (may exceed `storage.len()`).
    capacity_elems: usize,
    storage: Vec<f32>,
}

/// Owns all host buffer lifetimes for one runtime instance.
pub struct MemoryManager {
    next_id: u64,
    buffers: HashMap<u64, BufferEntry>,
    free_small: Vec<FreeBlock>,
    free_medium: Vec<FreeBlock>,
    free_large: Vec<FreeBlock>,
    /// Maximum charged host bytes (pool + Extra allocations).
    budget_bytes: usize,
    /// Currently charged host bytes.
    used_bytes: usize,
}

impl MemoryManager {
    /// Create a manager with the given host byte budget.
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

    /// Immutable lookup; fails if the ID was never allocated or already freed.
    pub fn get(&self, id: BufferId) -> Result<&BufferEntry> {
        self.buffers
            .get(&id.0)
            .ok_or(Error::Invalid("buffer does not exist"))
    }

    /// Mutable lookup; same failure mode as [`Self::get`].
    pub fn get_mut(&mut self, id: BufferId) -> Result<&mut BufferEntry> {
        self.buffers
            .get_mut(&id.0)
            .ok_or(Error::Invalid("buffer does not exist"))
    }

    /// Logical length of a buffer in `f32` elements.
    pub fn len_of(&self, id: BufferId) -> Result<usize> {
        Ok(self.get(id)?.len)
    }

    /// Copy `data` into a new buffer (or recycled freelist block).
    ///
    /// Marks `host_dirty` so a CUDA backend will H2D on first use. Empty slices
    /// are rejected. Updates pool hit/miss counters on `metrics`.
    pub fn upload(&mut self, data: &[f32], metrics: &mut RuntimeMetrics) -> Result<BufferId> {
        if data.is_empty() {
            return Err(Error::Invalid("upload: empty slice"));
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

    /// Allocate a zero-filled buffer of `len` elements (same dirty semantics as upload).
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

    /// Clone host contents when coherent and not in-flight.
    ///
    /// Returns [`Error::NotReady`] if jobs still hold the buffer or if the device
    /// copy is newer (`device_dirty`) — the runtime must call backend
    /// `prepare_download` first.
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

    /// Increment in-flight refcounts for every buffer touched by a submitted job.
    pub fn retain(&mut self, ids: &[BufferId]) -> Result<()> {
        for id in ids {
            self.get_mut(*id)?.in_flight += 1;
        }
        Ok(())
    }

    /// Decrement in-flight refcounts after a job completes (saturating).
    pub fn release(&mut self, ids: &[BufferId]) {
        for id in ids {
            if let Ok(entry) = self.get_mut(*id) {
                entry.in_flight = entry.in_flight.saturating_sub(1);
            }
        }
    }

    /// Destroy a buffer and return its storage to the freelist (or drop Extra charge).
    ///
    /// Fails with [`Error::Busy`] while `in_flight > 0`.
    pub fn free(&mut self, id: BufferId) -> Result<()> {
        let entry = self
            .buffers
            .remove(&id.0)
            .ok_or(Error::Invalid("buffer does not exist"))?;
        if entry.in_flight > 0 {
            // Put it back so the caller can retry after drain.
            self.buffers.insert(id.0, entry);
            return Err(Error::Busy);
        }
        self.recycle(entry.class, entry.host);
        Ok(())
    }

    /// Pull storage from the matching freelist or allocate under budget.
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
                // Extra never hits freelists; charge exact need.
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

    /// Return storage to the freelist (or uncharge Extra).
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

    /// Charge `bytes` against the budget or return [`Error::Oom`].
    fn charge(&mut self, bytes: usize) -> Result<()> {
        if self.used_bytes + bytes > self.budget_bytes {
            return Err(Error::Oom);
        }
        self.used_bytes += bytes;
        Ok(())
    }
}
