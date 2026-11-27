//! Apple Silicon AMX (Matrix Coprocessor) tile plan and `.inst` opcode definitions.
//!
//! By default this module exposes only the pure-integer tile geometry plan, raw
//! 32-bit Apple AMX instruction encodings, and an honest status string reporting
//! that the AMX backend is designed/opt-in. Hardware `.inst` execution is gated
//! behind `#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "apple-amx"))]`
//! and is NEVER invoked in default builds.

/// Default AMX matrix tile shape (`32 x 32 x 32`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AmxTilePlan {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl AmxTilePlan {
    /// Standard Apple Silicon AMX tile geometry (`32 x 32 x 32`).
    pub const DEFAULT: Self = Self {
        tile_m: 32,
        tile_n: 32,
        tile_k: 32,
    };

    /// Construct an [`AmxTilePlan`] with explicit tile dimensions.
    #[inline]
    pub const fn new(tile_m: usize, tile_n: usize, tile_k: usize) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
        }
    }

    /// Returns the default `32 x 32 x 32` AMX tile plan.
    #[inline]
    pub const fn default_plan() -> Self {
        Self::DEFAULT
    }
}

impl Default for AmxTilePlan {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Base instruction word for Apple AMX coprocessor encodings (`0x0020_1000 | (op << 5) | operand`).
pub const AMX_BASE_OPCODE: u32 = 0x0020_1000;

/// Encode an Apple AMX `.inst` 32-bit opcode word from a 5-bit operation code and 5-bit operand.
#[inline]
pub const fn amx_encode_inst(op: u32, operand: u32) -> u32 {
    AMX_BASE_OPCODE | ((op & 0x1F) << 5) | (operand & 0x1F)
}

/// Apple AMX `AMX_SET` (`op = 17, imm5 = 0`): initialize/enable AMX register state.
pub const AMX_SET: u32 = amx_encode_inst(17, 0);

/// Apple AMX `AMX_CLR` (`op = 17, imm5 = 1`): clear/disable AMX register state.
pub const AMX_CLR: u32 = amx_encode_inst(17, 1);

/// Apple AMX `AMX_LDX` (`op = 0`): load X tile register from memory.
pub const AMX_LDX: u32 = amx_encode_inst(0, 0);

/// Apple AMX `AMX_LDY` (`op = 1`): load Y tile register from memory.
pub const AMX_LDY: u32 = amx_encode_inst(1, 0);

/// Apple AMX `AMX_STZ` (`op = 5`): store Z accumulator tile register to memory.
pub const AMX_STZ: u32 = amx_encode_inst(5, 0);

/// Apple AMX `AMX_MAC16` (`op = 14`): 16-bit integer multiply-accumulate into Z tile.
pub const AMX_MAC16: u32 = amx_encode_inst(14, 0);

/// Honest runtime status of the Apple AMX backend.
#[inline]
pub fn amx_backend_status() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "apple-amx"))]
    {
        "apple-amx opt-in enabled (.inst probe available)"
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos", feature = "apple-amx")))]
    {
        "designed/opt-in (enable feature apple-amx)"
    }
}

/// Opt-in Apple Silicon AMX hardware state enable/clear probe (`AMX_SET` followed by `AMX_CLR`).
///
/// # Safety
/// Executes proprietary Apple Silicon AMX `.inst` coprocessor instructions (`0x00201220` and
/// `0x00201221`), preceded by 3 `nop` instructions per Apple Silicon AMX calling conventions.
/// Caller must ensure execution occurs on an Apple Silicon macOS target that supports EL0 AMX.
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "apple-amx"))]
pub unsafe fn amx_set_clr_probe() {
    // SAFETY: Caller guarantees an Apple Silicon macOS host with AMX coprocessor access.
    // `AMX_SET` enables AMX state and `AMX_CLR` cleanly tears it down before returning.
    unsafe {
        core::arch::asm!(
            "nop",
            "nop",
            "nop",
            ".inst 0x00201220",
            "nop",
            "nop",
            "nop",
            ".inst 0x00201221",
            options(nostack)
        );
    }
}
