//! Compute graph initialization plan grounded in [`BumpAllocator`] capacity and alignment.

use crate::ExecutionMode;
use crate::allocator::{BumpAllocator, CACHE_LINE_BYTES};

/// Verified initialization plan emitted by [`BigBang::ignite_tensor_graph`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BigBangPlan {
    pub arena_capacity_bytes: usize,
    pub alignment_bytes: usize,
    pub active_layers: u32,
    pub epoch: u64,
    pub mode: ExecutionMode,
}

/// Graph instantiation entry point recording arena geometry.
pub struct BigBang;

impl BigBang {
    /// Inspect `allocator` and emit a verified initialization plan for `active_layers`.
    #[inline]
    pub fn ignite_tensor_graph(allocator: &BumpAllocator, active_layers: u32) -> BigBangPlan {
        BigBangPlan {
            arena_capacity_bytes: allocator.capacity_bytes(),
            alignment_bytes: CACHE_LINE_BYTES,
            active_layers,
            epoch: allocator.epoch(),
            mode: ExecutionMode::Measured,
        }
    }
}
