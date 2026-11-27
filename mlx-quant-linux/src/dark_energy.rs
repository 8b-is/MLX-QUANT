//! Pure-rational (`Ratio { p, q }`) entropy / temperature scaling with zero floating point.

use crate::ExecutionMode;
use mlx_quant_core::ratio::Ratio;

/// Exact rational entropy scaler preventing repetitive generation collapse.
pub struct DarkEnergy;

impl DarkEnergy {
    /// Scale `base` entropy by exact rational `repetition` (`Ratio::mul_exact`).
    ///
    /// Preserves `base` when `repetition == Ratio::ZERO` so entropy never collapses to zero.
    #[inline]
    pub const fn accelerate_entropy(base: Ratio, repetition: Ratio) -> Ratio {
        if repetition.p == 0 || repetition.q == 0 {
            return base;
        }
        base.mul_exact(repetition)
    }

    /// Dampen `base` entropy by exact rational `damping` (`Ratio::div_exact`).
    ///
    /// Preserves `base` when `damping == Ratio::ZERO`.
    #[inline]
    pub const fn dampen_entropy(base: Ratio, damping: Ratio) -> Ratio {
        if damping.p == 0 || damping.q == 0 {
            return base;
        }
        base.div_exact(damping)
    }

    /// Audit classification (exact integer `Ratio` arithmetic is measured on host).
    #[inline]
    pub const fn execution_mode() -> ExecutionMode {
        ExecutionMode::Measured
    }
}
