//! Fixed-capacity scatter-gather descriptor list (zero `Vec`).

use crate::{ExecutionMode, fnv1a_64};

/// Maximum scatter-gather segments tracked per fixed descriptor list.
pub const MAX_SG_SEGMENTS: usize = 16;

/// Single contiguous memory segment `(base_addr, len_bytes)`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScatterGatherSegment {
    pub base_addr: usize,
    pub len_bytes: usize,
}

/// Fixed-array scatter-gather descriptor table (`<= 16` segments, zero heap allocations).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScatterGatherList {
    pub segments: [ScatterGatherSegment; MAX_SG_SEGMENTS],
    pub count: usize,
    pub total_bytes: usize,
    pub mode: ExecutionMode,
}

/// Scatter-gather segment assembler for fragmented tensor payloads.
pub struct AsteroidBelt;

impl AsteroidBelt {
    /// Assemble up to `MAX_SG_SEGMENTS` `(base_addr, len_bytes)` fragments into a fixed list.
    pub fn gather_fragments(fragments: &[(usize, usize)]) -> ScatterGatherList {
        let count = fragments.len().min(MAX_SG_SEGMENTS);
        let mut segments = [ScatterGatherSegment::default(); MAX_SG_SEGMENTS];
        let mut total_bytes = 0usize;
        for (idx, &(base_addr, len_bytes)) in fragments[..count].iter().enumerate() {
            segments[idx] = ScatterGatherSegment {
                base_addr,
                len_bytes,
            };
            total_bytes = total_bytes.saturating_add(len_bytes);
        }
        ScatterGatherList {
            segments,
            count,
            total_bytes,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }

    /// Gather live host byte slices into a [`ScatterGatherList`] and compute their combined FNV-1a digest.
    pub fn gather_slices(slices: &[&[u8]]) -> (ScatterGatherList, u64) {
        let count = slices.len().min(MAX_SG_SEGMENTS);
        let mut segments = [ScatterGatherSegment::default(); MAX_SG_SEGMENTS];
        let mut total_bytes = 0usize;
        let mut combined_digest = 0xCBF2_9CE4_8422_2325u64;
        for (idx, &slice) in slices[..count].iter().enumerate() {
            segments[idx] = ScatterGatherSegment {
                base_addr: slice.as_ptr() as usize,
                len_bytes: slice.len(),
            };
            total_bytes = total_bytes.saturating_add(slice.len());
            combined_digest ^= fnv1a_64(slice);
            combined_digest = combined_digest.wrapping_mul(0x0100_0000_01B3);
        }
        (
            ScatterGatherList {
                segments,
                count,
                total_bytes,
                mode: ExecutionMode::Measured,
            },
            combined_digest,
        )
    }
}
