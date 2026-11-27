//! Epoch reclamation wrapping [`BumpAllocator::reset_epoch`] with `&mut self` safety.

use crate::allocator::BumpAllocator;

/// Epoch-scoped arena reclamation ("Hawking radiation" eviction of prior-epoch allocations).
pub struct HawkingRadiation;

impl HawkingRadiation {
    /// Reclaim the current epoch by resetting `allocator`'s bump cursor via exclusive `&mut self`.
    ///
    /// Returns `(reclaimed_bytes, new_epoch)`. Because `&mut BumpAllocator` is required, Rust's
    /// borrow checker statically prevents calling this while any [`crate::tensor::Tensor`] borrow lives.
    #[inline]
    pub fn evaporate_cold_cache(allocator: &mut BumpAllocator) -> (usize, u64) {
        let reclaimed_bytes = allocator.used_bytes();
        allocator.reset_epoch();
        (reclaimed_bytes, allocator.epoch())
    }
}
