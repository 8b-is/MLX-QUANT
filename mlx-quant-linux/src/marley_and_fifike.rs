//! Runtime arena & bounds firewall integrity guardian (`Fifike` & `Marley`).

use crate::ExecutionMode;
use crate::allocator::BumpAllocator;
use crate::event_horizon::EventHorizon;
use mlx_quant_core::ratio::splitmix64;

/// Verified runtime integrity report produced by [`TheGuardians::verify_integrity`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GuardianStatus {
    pub fifike_purr_hz_range: (u16, u16),
    pub marley_integrity_digest: u64,
    pub active_slots_mask: u64,
    pub used_bytes: usize,
    pub capacity_bytes: usize,
    pub healthy: bool,
    pub mode: ExecutionMode,
}

/// Runtime safety & coherence guardian checking arena invariants and firewall bounds.
pub struct TheGuardians;

impl TheGuardians {
    /// Verify `allocator` consistency (`used <= capacity`) and confirm its backing buffer falls
    /// cleanly inside `horizon`, computing a deterministic SplitMix64 integrity digest.
    pub fn verify_integrity(allocator: &BumpAllocator, horizon: &EventHorizon) -> GuardianStatus {
        let used_bytes = allocator.used_bytes();
        let capacity_bytes = allocator.capacity_bytes();
        let active_slots_mask = allocator.active_slot_mask();
        let (base, cap) = allocator.backing_addr_range();
        let bounds_ok = cap == 0 || horizon.contains(base, cap);
        let healthy = used_bytes <= capacity_bytes && bounds_ok;

        let mut digest = splitmix64((used_bytes as u64) ^ ((capacity_bytes as u64) << 17));
        digest = splitmix64(digest ^ active_slots_mask ^ allocator.epoch());
        digest = splitmix64(digest ^ (base as u64) ^ u64::from(healthy));

        GuardianStatus {
            fifike_purr_hz_range: (25, 150),
            marley_integrity_digest: digest,
            active_slots_mask,
            used_bytes,
            capacity_bytes,
            healthy,
            mode: ExecutionMode::Measured,
        }
    }

    /// Fifike harmonic baseline purr range (`25..=150` Hz).
    #[inline]
    pub const fn fifike_universal_heal() -> (u16, u16) {
        (25, 150)
    }

    /// Marley bounds-watch guard confirming `horizon` end address does not overflow.
    #[inline]
    pub const fn marley_human_protect(horizon: &EventHorizon) -> bool {
        horizon.end_addr().is_some()
    }
}
