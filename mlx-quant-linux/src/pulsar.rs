//! Hardware counter synchronization and monotonic nanosecond elapsed timing.
//!
//! Backed by `mlx_quant_asm::hardware_counter()` (`cntvct_el0` on AArch64, `rdtsc` on x86_64)
//! and `mlx_quant_asm::hardware_counter_freq_hz()` (`cntfrq_el0` on AArch64).

use crate::ExecutionMode;

/// Hardware timing beacon snapshot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PulsarBeacon {
    pub ticks: u64,
    pub freq_hz: u64,
    pub mode: ExecutionMode,
}

/// Pure-integer hardware counter timer and beacon emitter.
pub struct Pulsar;

impl Pulsar {
    /// Sample the architecture's unprivileged monotonic hardware counter (`cntvct_el0` / `rdtsc`).
    #[inline]
    pub fn now_ticks() -> u64 {
        mlx_quant_asm::hardware_counter()
    }

    /// Read the hardware counter frequency in Hz when architecturally exposed (`cntfrq_el0`).
    #[inline]
    pub fn freq_hz() -> u64 {
        mlx_quant_asm::hardware_counter_freq_hz()
    }

    /// Emit a synchronization beacon carrying current hardware ticks and frequency.
    #[inline]
    pub fn emit_sync_beacon() -> PulsarBeacon {
        let freq_hz = Self::freq_hz();
        let ticks = Self::now_ticks();
        let mode = if freq_hz > 0 {
            ExecutionMode::Measured
        } else {
            ExecutionMode::DesignedDescriptor
        };
        PulsarBeacon {
            ticks,
            freq_hz,
            mode,
        }
    }

    /// Convert two counter samples `(t0, t1)` into elapsed nanoseconds using `Self::freq_hz()`.
    ///
    /// Returns `0` cleanly if `t1 <= t0` or `freq_hz == 0`.
    #[inline]
    pub fn elapsed_ns(t0: u64, t1: u64) -> u64 {
        Self::elapsed_ns_with_freq(t0, t1, Self::freq_hz())
    }

    /// Pure-integer conversion of counter ticks `(t0, t1)` into nanoseconds for an explicit `freq_hz`.
    #[inline]
    pub const fn elapsed_ns_with_freq(t0: u64, t1: u64, freq_hz: u64) -> u64 {
        if freq_hz == 0 || t1 <= t0 {
            return 0;
        }
        let delta = (t1 - t0) as u128;
        let ns = (delta.saturating_mul(1_000_000_000u128)) / (freq_hz as u128);
        if ns > (u64::MAX as u128) {
            u64::MAX
        } else {
            ns as u64
        }
    }
}
