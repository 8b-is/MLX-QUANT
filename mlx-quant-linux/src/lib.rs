//! `mlx-quant-linux`: Pure-integer, zero-float bare-metal execution substrate for MLX-QUANT.
//!
//! Connects `mlx-quant-core` (`#![no_std]` rational/ternary/BeeHive oracles and lock-free epoch
//! bump arenas), `mlx-quant-asm` (AArch64 NEON/DotProd & x86_64 AVX2/AVX-512 inline-ASM/SIMD
//! kernels + `cntvct_el0` + `prfm`), and `mlx-quant-gpu` (live Apple Metal `.metallib` runtime
//! + 64-byte aligned AMD CDNA3 Type-3 PM4 ring).

pub use mlx_quant_asm;
pub use mlx_quant_core;
pub use mlx_quant_gpu;

pub mod agx_doorbell;
pub mod allocator;
pub mod amd_mi300x;
pub mod antigravity;
pub mod asteroid_belt;
pub mod big_bang;
pub mod cmb;
pub mod dark_energy;
pub mod dark_matter;
pub mod event_horizon;
pub mod fusion_polarity;
pub mod gravitational_lensing;
pub mod hawking_radiation;
pub mod io_blackhole;
pub mod magnetar;
pub mod marley_and_fifike;
pub mod math;
pub mod multiverse;
pub mod polar_queue;
pub mod pulsar;
pub mod quantum_entanglement;
pub mod quasar;
pub mod redshift;
pub mod supernova;
pub mod tensor;
pub mod time_dilation;
pub mod white_hole;
pub mod wormhole;

#[cfg(test)]
mod tests;

/// Honest provenance classification for every subsystem in `mlx-quant-linux`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Executed live on host CPU/NEON/Metal hardware and verified bit-for-bit.
    Measured,
    /// Pure-integer architectural descriptor / ring layout for hardware not present on host.
    DesignedDescriptor,
}

impl ExecutionMode {
    /// Human-readable audit label.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "Measured",
            Self::DesignedDescriptor => "DesignedDescriptor",
        }
    }
}

impl core::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shared 64-bit FNV-1a digest helper for deterministic byte-level integrity checks.
#[inline]
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01B3;
    let mut h = OFFSET_BASIS;
    let mut i = 0usize;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(PRIME);
        i += 1;
    }
    h
}

/// Copy up to `N` UTF-8/ASCII bytes into a fixed stack buffer, returning `(buf, len)`.
#[inline]
pub(crate) fn copy_fixed_label<const N: usize>(src: &str) -> ([u8; N], usize) {
    let mut buf = [0u8; N];
    let bytes = src.as_bytes();
    let len = bytes.len().min(N);
    buf[..len].copy_from_slice(&bytes[..len]);
    (buf, len)
}
