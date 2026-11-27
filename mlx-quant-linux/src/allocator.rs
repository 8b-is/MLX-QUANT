//! Heap-backed, 64-byte aligned, lock-free epoch bump allocator.
//!
//! Permanently resolves the RC1 `static BUMP_HEAP` + `Cell` + `unsafe impl Sync` aliasing hazard:
//! - Backing memory is heap-allocated with 64-byte alignment and owned cleanly via `Drop`.
//! - Concurrent `alloc_bytes(&self, ...)` calls execute lock-free `AtomicUsize` CAS inside
//!   [`EpochBumpArenaSlice`], partitioning strictly disjoint sub-regions.
//! - [`BumpAllocator::reset_epoch`] requires `&mut self`, statically preventing use-after-free
//!   or aliasing across threads and live [`crate::tensor::Tensor`] borrows.

use core::alloc::Layout;
use core::ptr::NonNull;
use mlx_quant_core::arena::EpochBumpArenaSlice;

/// Default allocator capacity (`64 MiB`), matching the RC1 tensor arena budget on the heap.
pub const DEFAULT_BUMP_CAPACITY_BYTES: usize = 64 * 1024 * 1024;
/// Minimum cache-line alignment (`64` bytes) for every tensor allocation.
pub const CACHE_LINE_BYTES: usize = 64;

/// Thread-safe (`Send + Sync`) lock-free epoch bump allocator backed by a 64-byte aligned heap buffer.
pub struct BumpAllocator {
    arena: EpochBumpArenaSlice<'static>,
    backing: Option<(NonNull<u8>, Layout)>,
}

// SAFETY: `BumpAllocator` exclusively owns the heap buffer described by `backing`, freeing it
// only in `Drop(&mut self)`. All shared `&self` allocations delegate to `EpochBumpArenaSlice`,
// which uses lock-free CAS to yield strictly disjoint `&mut [u8]` sub-slices lifetime-bound to
// `&self`. Both `reset_epoch(&mut self)` and `Drop(&mut self)` require exclusive access.
unsafe impl Send for BumpAllocator {}
// SAFETY: Same lock-free disjoint CAS partitioning invariant as `Send` above.
unsafe impl Sync for BumpAllocator {}

impl core::fmt::Debug for BumpAllocator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BumpAllocator")
            .field("used_bytes", &self.used_bytes())
            .field("capacity_bytes", &self.capacity_bytes())
            .field("epoch", &self.epoch())
            .field("high_water_bytes", &self.high_water_bytes())
            .field("alloc_count", &self.alloc_count())
            .field("active_slot_mask", &self.active_slot_mask())
            .finish()
    }
}

impl Default for BumpAllocator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BumpAllocator {
    /// Create a heap-backed 64-byte aligned allocator with the default `64 MiB` capacity.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUMP_CAPACITY_BYTES)
    }

    /// Create a heap-backed 64-byte aligned allocator with explicit capacity in bytes.
    pub fn with_capacity(bytes: usize) -> Self {
        if bytes == 0 {
            return Self {
                arena: EpochBumpArenaSlice::new(&mut []),
                backing: None,
            };
        }

        let layout =
            Layout::from_size_align(bytes, CACHE_LINE_BYTES).expect("invalid BumpAllocator size");
        // SAFETY: `bytes > 0` and `CACHE_LINE_BYTES` (64) is a non-zero power of two.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let Some(nn) = NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        // SAFETY: `nn` points to `bytes` zeroed, 64-byte aligned bytes exclusively owned until `Drop`.
        // Returned sub-slices from `alloc_bytes(&self, ...)` carry the `&self` lifetime, never `'static`.
        let backing_slice: &'static mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(nn.as_ptr(), bytes) };
        let arena = EpochBumpArenaSlice::new(backing_slice);
        Self {
            arena,
            backing: Some((nn, layout)),
        }
    }

    /// Lock-free bump-allocate `size` bytes aligned to `align.max(64)` (power of two if `> 64`).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<&mut [u8]> {
        self.arena.alloc_bytes(size, align)
    }

    /// Lock-free bump-allocate `size` bytes aligned to `64`-byte cache lines for tensor storage.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_tensor_bytes(&self, size: usize) -> Option<&mut [u8]> {
        self.arena.alloc_bytes(size, CACHE_LINE_BYTES)
    }

    /// Lock-free allocate and default-initialize `&mut [T]` aligned to `align_of::<T>().max(64)`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_slice_mut<T: Copy + Default>(&self, len: usize) -> Option<&mut [T]> {
        self.arena.alloc_slice_mut::<T>(len)
    }

    /// Reset the bump cursor to `0` and increment the epoch counter.
    ///
    /// Requiring `&mut self` statically guarantees no borrowed slices or `Tensor<'a>` values
    /// referencing this arena remain alive across epoch reclamation.
    #[inline]
    pub fn reset_epoch(&mut self) {
        self.arena.reset_epoch();
    }

    /// Bytes currently allocated in the active epoch.
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.arena.used_bytes()
    }

    /// Total byte capacity of the owned heap buffer.
    #[inline]
    pub const fn capacity_bytes(&self) -> usize {
        self.arena.capacity_bytes()
    }

    /// Monotonic epoch counter incremented on each [`Self::reset_epoch`].
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.arena.epoch()
    }

    /// Peak `used_bytes` observed across all epochs.
    #[inline]
    pub fn high_water_bytes(&self) -> usize {
        self.arena.high_water_bytes()
    }

    /// Cumulative count of successful allocations across epochs.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.arena.alloc_count()
    }

    /// Atomically claim one of 64 concurrent tensor tracking slots (`0..64`).
    #[inline]
    pub fn claim_slot(&self) -> Option<u8> {
        self.arena.claim_slot()
    }

    /// Atomically release a previously claimed slot ID; returns `true` if it was active.
    #[inline]
    pub fn release_slot(&self, slot: u8) -> bool {
        self.arena.release_slot(slot)
    }

    /// Snapshot bitmask of active slot IDs.
    #[inline]
    pub fn active_slot_mask(&self) -> u64 {
        self.arena.active_slot_mask()
    }

    /// Return `(base_address, capacity_bytes)` of the owned backing buffer for firewall verification.
    #[inline]
    pub fn backing_addr_range(&self) -> (usize, usize) {
        match self.backing {
            Some((ptr, _)) => (ptr.as_ptr() as usize, self.arena.capacity_bytes()),
            None => (0, 0),
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        if let Some((ptr, layout)) = self.backing.take() {
            // SAFETY: `ptr` was allocated with `alloc_zeroed(layout)` in `with_capacity`,
            // and `&mut self` guarantees no live references into the arena exist.
            unsafe {
                std::alloc::dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}
