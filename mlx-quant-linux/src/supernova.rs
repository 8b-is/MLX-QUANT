//! Honest DVFS / SMC power-cap descriptor (disabled by default in userspace, per RC1 audit).

use crate::ExecutionMode;

/// Pure-integer DVFS burst policy descriptor (`thermal_cap_millic` in thousandths of deg C).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DvfsBurstPolicy {
    pub enabled: bool,
    pub target_power_mw: u32,
    pub thermal_cap_millic: i32,
    pub mode: ExecutionMode,
}

/// SMC / DVFS burst policy controller (privileged kernel interface is `DesignedDescriptor`).
pub struct Supernova;

impl Supernova {
    /// Default RC1-compliant policy: TDP uncap disabled in unprivileged userspace.
    #[inline]
    pub const fn default_safe_policy() -> DvfsBurstPolicy {
        DvfsBurstPolicy {
            enabled: false,
            target_power_mw: 0,
            thermal_cap_millic: 95_000,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }

    /// Record a requested power/thermal envelope without modifying host SMC registers (`enabled: false`).
    #[inline]
    pub const fn trigger_burst(target_power_mw: u32, thermal_cap_millic: i32) -> DvfsBurstPolicy {
        DvfsBurstPolicy {
            enabled: false,
            target_power_mw,
            thermal_cap_millic,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
