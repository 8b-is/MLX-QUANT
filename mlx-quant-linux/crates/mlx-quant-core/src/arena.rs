//! Lock-free epoch-scoped bump arenas (`#![no_std]`, 64-byte cache-line aligned).
//!
//! Permanently resolves the legacy `BumpAllocator` `unsafe impl Sync` / `Cell` data-race
//! hazard:
//! - Concurrent `alloc_bytes(&self, ...)` calls execute lock-free CAS (`fetch_update(AcqRel, Acquire)`)
//!   returning strictly disjoint sub-regions aligned to `align.max(64)` (power of two).
//! - Returned `&mut [u8]` / `&mut [T]` slices tie their lifetime to `&self`, so calling
//!   `reset_epoch(&mut self)` is statically prevented while any allocated slice borrow is alive.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache-line aligned backing storage (`64`-byte aligned) for [`EpochBumpArena`].
#[repr(C, align(64))]
pub struct AlignedStorage<const N: usize>(UnsafeCell<[u8; N]>);

impl<const N: usize> AlignedStorage<N> {
    /// Zero-initialized 64-byte aligned static/stack backing array.
    #[inline]
    pub const fn new() -> Self {
        Self(UnsafeCell::new([0u8; N]))
    }
}

impl<const N: usize> Default for AlignedStorage<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Shared lock-free bump cursor, epoch counter, slot bitmap, and telemetry.
struct BumpCore {
    offset: AtomicUsize,
    epoch: AtomicU64,
    slot_bitmap: AtomicU64,
    high_water_bytes: AtomicUsize,
    alloc_count: AtomicUsize,
}

impl BumpCore {
    #[inline]
    const fn new() -> Self {
        Self {
            offset: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
            slot_bitmap: AtomicU64::new(0),
            high_water_bytes: AtomicUsize::new(0),
            alloc_count: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn reserve_range(
        &self,
        base_addr: usize,
        capacity: usize,
        size: usize,
        align: usize,
    ) -> Option<usize> {
        let eff_align = if align <= 64 {
            64
        } else if align.is_power_of_two() {
            align
        } else {
            return None;
        };
        let mask = eff_align - 1;
        let prev = self
            .offset
            .try_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let addr = base_addr.checked_add(cur)?;
                let aligned_addr = addr.checked_add(mask)? & !mask;
                let start = aligned_addr - base_addr;
                let end = start.checked_add(size)?;
                if end > capacity {
                    return None;
                }
                Some(end)
            })
            .ok()?;

        let aligned_addr = (base_addr + prev + mask) & !mask;
        let start = aligned_addr - base_addr;
        let end = start + size;
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.high_water_bytes.fetch_max(end, Ordering::Relaxed);
        Some(start)
    }

    #[inline]
    fn reset_epoch(&mut self) {
        *self.offset.get_mut() = 0;
        *self.epoch.get_mut() = self.epoch.get_mut().wrapping_add(1);
    }

    #[inline]
    fn used_bytes(&self) -> usize {
        self.offset.load(Ordering::Acquire)
    }

    #[inline]
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    #[inline]
    fn high_water_bytes(&self) -> usize {
        self.high_water_bytes.load(Ordering::Acquire)
    }

    #[inline]
    fn alloc_count(&self) -> usize {
        self.alloc_count.load(Ordering::Acquire)
    }

    #[inline]
    fn claim_slot(&self) -> Option<u8> {
        let mut claimed = 0u8;
        self.slot_bitmap
            .try_update(Ordering::AcqRel, Ordering::Acquire, |mask| {
                if mask == u64::MAX {
                    return None;
                }
                let bit = (!mask).trailing_zeros() as u8;
                claimed = bit;
                Some(mask | (1u64 << bit))
            })
            .ok()?;
        Some(claimed)
    }

    #[inline]
    fn release_slot(&self, slot: u8) -> bool {
        if slot >= 64 {
            return false;
        }
        let bit = 1u64 << slot;
        let prev = self.slot_bitmap.fetch_and(!bit, Ordering::AcqRel);
        (prev & bit) != 0
    }

    #[inline]
    fn active_slot_mask(&self) -> u64 {
        self.slot_bitmap.load(Ordering::Acquire)
    }
}

/// Align a caller-provided mutable byte buffer up to a `64`-byte boundary.
///
/// Returns an empty slice if fewer than `64 - (ptr % 64)` bytes exist.
#[inline]
pub fn align_subslice_64(buf: &mut [u8]) -> &mut [u8] {
    let base = buf.as_ptr() as usize;
    let rem = base & 63;
    let skip = if rem == 0 { 0 } else { 64 - rem };
    if skip >= buf.len() {
        &mut []
    } else {
        &mut buf[skip..]
    }
}

/// Initialize a freshly carved byte range into `&mut [T]` with `T::default()`.
///
/// # Safety
/// `base_ptr` must be valid for writes across `start..start + len * size_of::<T>()`,
/// disjoint from all other live references, and aligned to `align_of::<T>()`.
#[inline]
unsafe fn init_slice_from_raw<'a, T: Copy + Default>(
    base_ptr: *mut u8,
    start: usize,
    len: usize,
) -> &'a mut [T] {
    if size_of::<T>() == 0 {
        // SAFETY: Zero-sized types require no backing bytes; `NonNull::dangling()` is aligned.
        return unsafe { core::slice::from_raw_parts_mut(NonNull::<T>::dangling().as_ptr(), len) };
    }
    // SAFETY: Caller verified `start + len * size_of::<T>() <= capacity` and exclusive ownership.
    let typed_ptr = unsafe { base_ptr.add(start) } as *mut T;
    let mut i = 0usize;
    while i < len {
        // SAFETY: Slot `i` is within the exclusively carved region and aligned for `T`.
        unsafe { typed_ptr.add(i).write(T::default()) };
        i += 1;
    }
    // SAFETY: All `len` elements are initialized and uniquely owned for lifetime `'a`.
    unsafe { core::slice::from_raw_parts_mut(typed_ptr, len) }
}

/// Fixed-capacity `const`-constructible lock-free epoch bump arena (`#![no_std]`).
pub struct EpochBumpArena<const N: usize> {
    storage: AlignedStorage<N>,
    core: BumpCore,
}

// SAFETY: Concurrent callers mutate `core.offset` via lock-free `AcqRel`/`Acquire` CAS,
// obtaining strictly disjoint regions in `storage.0`; `reset_epoch(&mut self)` requires
// exclusive access, preventing any concurrent or overlapping mutable borrows.
unsafe impl<const N: usize> Sync for EpochBumpArena<N> {}

impl<const N: usize> Default for EpochBumpArena<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> EpochBumpArena<N> {
    /// Create a new empty arena (`const`-evaluable for static `#![no_std]` placement).
    #[inline]
    pub const fn new() -> Self {
        Self {
            storage: AlignedStorage::new(),
            core: BumpCore::new(),
        }
    }

    /// Allocate `size` bytes aligned to at least `align.max(64)` (must be a power of two if `> 64`).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<&mut [u8]> {
        let base_ptr = self.storage.0.get() as *mut u8;
        let start = self.core.reserve_range(base_ptr as usize, N, size, align)?;
        // SAFETY: `start..start + size` lies inside `[0..N]` and is uniquely claimed via CAS
        // until `&self` is dropped or exclusively borrowed by `reset_epoch(&mut self)`.
        Some(unsafe { core::slice::from_raw_parts_mut(base_ptr.add(start), size) })
    }

    /// Allocate and zero/default-initialize a typed slice `&mut [T]` aligned to `align_of::<T>().max(64)`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_slice_mut<T: Copy + Default>(&self, len: usize) -> Option<&mut [T]> {
        let elem_size = size_of::<T>();
        let total_bytes = if elem_size == 0 {
            0
        } else {
            len.checked_mul(elem_size)?
        };
        let base_ptr = self.storage.0.get() as *mut u8;
        let start =
            self.core
                .reserve_range(base_ptr as usize, N, total_bytes, align_of::<T>())?;
        // SAFETY: `start..start + total_bytes` is exclusively reserved and `align_of::<T>()`-aligned.
        Some(unsafe { init_slice_from_raw(base_ptr, start, len) })
    }

    /// Reset the bump cursor to `0` and advance the epoch counter.
    #[inline]
    pub fn reset_epoch(&mut self) {
        self.core.reset_epoch();
    }

    /// Bytes currently allocated in the active epoch.
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.core.used_bytes()
    }

    /// Total byte capacity of the arena.
    #[inline]
    pub const fn capacity_bytes(&self) -> usize {
        N
    }

    /// Monotonic epoch counter incremented on each [`Self::reset_epoch`].
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.core.epoch()
    }

    /// Peak `used_bytes` observed across all epochs.
    #[inline]
    pub fn high_water_bytes(&self) -> usize {
        self.core.high_water_bytes()
    }

    /// Cumulative number of successful allocations.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.core.alloc_count()
    }

    /// Atomically claim one of 64 concurrent slot IDs (`0..64`).
    #[inline]
    pub fn claim_slot(&self) -> Option<u8> {
        self.core.claim_slot()
    }

    /// Atomically release a previously claimed slot ID; returns `true` if it was set.
    #[inline]
    pub fn release_slot(&self, slot: u8) -> bool {
        self.core.release_slot(slot)
    }

    /// Snapshot bitmask of currently active slots.
    #[inline]
    pub fn active_slot_mask(&self) -> u64 {
        self.core.active_slot_mask()
    }
}

/// Borrowed-buffer lock-free epoch bump arena for large heap-backed buffers (e.g., 64 MiB in `std`)
/// without overflowing thread stacks.
pub struct EpochBumpArenaSlice<'a> {
    base: NonNull<u8>,
    capacity: usize,
    core: BumpCore,
    _marker: PhantomData<&'a mut [u8]>,
}

// SAFETY: Exclusive provenance over the borrowed buffer for `'a`, partitioned lock-free via CAS
// into disjoint `&mut [u8]` sub-slices tied to `&self`, with `reset_epoch(&mut self)` exclusive.
unsafe impl<'a> Send for EpochBumpArenaSlice<'a> {}
// SAFETY: Same lock-free disjoint CAS partitioning invariant as `EpochBumpArena<N>`.
unsafe impl<'a> Sync for EpochBumpArenaSlice<'a> {}

impl<'a> EpochBumpArenaSlice<'a> {
    /// Wrap a caller-owned mutable buffer, aligning its usable base pointer up to `64` bytes.
    #[inline]
    pub fn new(backing: &'a mut [u8]) -> Self {
        let aligned = align_subslice_64(backing);
        let capacity = aligned.len();
        let base = NonNull::new(aligned.as_mut_ptr()).unwrap_or_else(NonNull::dangling);
        Self {
            base,
            capacity,
            core: BumpCore::new(),
            _marker: PhantomData,
        }
    }

    /// Convenience constructor borrowing an [`AlignedStorage`] buffer.
    #[inline]
    pub fn from_aligned_storage<const N: usize>(storage: &'a mut AlignedStorage<N>) -> Self {
        Self::new(storage.0.get_mut())
    }

    /// Allocate `size` bytes aligned to at least `align.max(64)` (must be a power of two if `> 64`).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<&mut [u8]> {
        let base_ptr = self.base.as_ptr();
        let start = self
            .core
            .reserve_range(base_ptr as usize, self.capacity, size, align)?;
        // SAFETY: `start..start + size` is within `[0..self.capacity]` and uniquely claimed via CAS.
        Some(unsafe { core::slice::from_raw_parts_mut(base_ptr.add(start), size) })
    }

    /// Allocate and default-initialize `&mut [T]` aligned to `align_of::<T>().max(64)`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn alloc_slice_mut<T: Copy + Default>(&self, len: usize) -> Option<&mut [T]> {
        let elem_size = size_of::<T>();
        let total_bytes = if elem_size == 0 {
            0
        } else {
            len.checked_mul(elem_size)?
        };
        let base_ptr = self.base.as_ptr();
        let start = self.core.reserve_range(
            base_ptr as usize,
            self.capacity,
            total_bytes,
            align_of::<T>(),
        )?;
        // SAFETY: `start..start + total_bytes` is exclusively reserved and aligned for `T`.
        Some(unsafe { init_slice_from_raw(base_ptr, start, len) })
    }

    /// Reset the bump cursor to `0` and advance the epoch counter.
    #[inline]
    pub fn reset_epoch(&mut self) {
        self.core.reset_epoch();
    }

    /// Bytes currently allocated in the active epoch.
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.core.used_bytes()
    }

    /// Usable 64-byte-aligned byte capacity of this slice arena.
    #[inline]
    pub const fn capacity_bytes(&self) -> usize {
        self.capacity
    }

    /// Monotonic epoch counter incremented on each [`Self::reset_epoch`].
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.core.epoch()
    }

    /// Peak `used_bytes` observed across all epochs.
    #[inline]
    pub fn high_water_bytes(&self) -> usize {
        self.core.high_water_bytes()
    }

    /// Cumulative number of successful allocations.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.core.alloc_count()
    }

    /// Atomically claim one of 64 concurrent slot IDs (`0..64`).
    #[inline]
    pub fn claim_slot(&self) -> Option<u8> {
        self.core.claim_slot()
    }

    /// Atomically release a previously claimed slot ID; returns `true` if it was set.
    #[inline]
    pub fn release_slot(&self, slot: u8) -> bool {
        self.core.release_slot(slot)
    }

    /// Snapshot bitmask of currently active slots.
    #[inline]
    pub fn active_slot_mask(&self) -> u64 {
        self.core.active_slot_mask()
    }
}
