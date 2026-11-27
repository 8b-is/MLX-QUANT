//! L1 cache-line prefetching (`prfm pldl1keep` on AArch64, `_mm_prefetch` on x86_64).

/// Cache-line stride in bytes for L1 data prefetching.
pub const CACHE_LINE_STRIDE: usize = 64;

/// Prefetches tensor memory regions into L1 cache in 64-byte strides via `mlx-quant-asm`.
pub struct Antigravity;

impl Antigravity {
    /// Issue non-faulting L1 prefetch hints across `[tensor_ptr .. tensor_ptr + size)`
    /// every `64` bytes (`step_by(64)`). Returns the number of cache lines prefetched.
    #[inline]
    pub fn levitate_tensor(tensor_ptr: *const u8, size: usize) -> usize {
        if tensor_ptr.is_null() || size == 0 {
            return 0;
        }
        let mut lines = 0usize;
        for offset in (0..size).step_by(CACHE_LINE_STRIDE) {
            mlx_quant_asm::prefetch_l1(tensor_ptr.wrapping_add(offset));
            lines += 1;
        }
        lines
    }

    /// Safe slice wrapper prefetching every 64-byte cache line in `bytes`.
    #[inline]
    pub fn levitate_slice(bytes: &[u8]) -> usize {
        if bytes.is_empty() {
            return 0;
        }
        let mut lines = 0usize;
        for offset in (0..bytes.len()).step_by(CACHE_LINE_STRIDE) {
            mlx_quant_asm::prefetch_l1(bytes[offset..].as_ptr());
            lines += 1;
        }
        lines
    }
}
