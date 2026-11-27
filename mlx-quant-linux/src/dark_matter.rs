//! Pure-integer zero-page deduplication descriptor for sparse MoE tensor mappings.

use crate::ExecutionMode;
use mlx_quant_core::ratio::Ratio;

/// Descriptor capturing virtual sparse tensor bytes vs physical zero-page size.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SparseZeroPageDescriptor {
    pub virtual_bytes: usize,
    pub physical_page_bytes: usize,
    pub deduplication_ratio: Ratio,
    pub mode: ExecutionMode,
}

/// Sparse zero-page virtual mapping planner (OS fault-handler page sharing is `DesignedDescriptor`).
pub struct DarkMatter;

impl DarkMatter {
    /// Compute the exact rational deduplication descriptor for `virtual_bytes` backed by a single
    /// physical zero page of `page_size` bytes.
    #[inline]
    pub const fn ghost_map_sparse_tensor(
        virtual_bytes: usize,
        page_size: usize,
    ) -> SparseZeroPageDescriptor {
        let eff_page = if page_size == 0 { 16_384 } else { page_size };
        SparseZeroPageDescriptor {
            virtual_bytes,
            physical_page_bytes: eff_page,
            deduplication_ratio: Ratio::new(virtual_bytes as u64, eff_page as u64),
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
