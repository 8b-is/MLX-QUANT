//! Overflow-checked address-range firewall isolating tensor arena bounds.

use crate::ExecutionMode;

/// Contiguous address firewall `[base .. base + size)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EventHorizon {
    pub base: usize,
    pub size: usize,
}

impl EventHorizon {
    /// Construct a bounds firewall covering `[base .. base + size)`.
    #[inline]
    pub const fn erect_firewall(base: usize, size: usize) -> Self {
        Self { base, size }
    }

    /// Returns `true` iff `[addr .. addr + len)` lies completely inside `[self.base .. self.base + self.size)`
    /// without integer wraparound.
    #[inline]
    pub const fn contains(&self, addr: usize, len: usize) -> bool {
        let Some(horizon_end) = self.base.checked_add(self.size) else {
            return false;
        };
        let Some(req_end) = addr.checked_add(len) else {
            return false;
        };
        addr >= self.base && req_end <= horizon_end
    }

    /// Returns `true` iff the entire byte slice falls within this firewall.
    #[inline]
    pub fn contains_slice(&self, slice: &[u8]) -> bool {
        self.contains(slice.as_ptr() as usize, slice.len())
    }

    /// Checked exclusive upper bound `base + size`.
    #[inline]
    pub const fn end_addr(&self) -> Option<usize> {
        self.base.checked_add(self.size)
    }

    /// Audit classification (software bounds firewall is measured; hardware MMU traps are OS-gated).
    #[inline]
    pub const fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::Measured
    }
}
