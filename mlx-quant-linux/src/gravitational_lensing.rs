//! Fixed-array speculative token drafting and exact prefix verification (zero `Vec`).

use crate::ExecutionMode;
use mlx_quant_core::ratio::{Ratio, splitmix64};

/// Maximum speculative lookahead draft horizon per step.
pub const MAX_SPEC_DRAFT: usize = 8;

/// Stack-allocated speculative draft token sequence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpeculativeDraft {
    pub tokens: [u32; MAX_SPEC_DRAFT],
    pub len: usize,
}

impl SpeculativeDraft {
    /// Active drafted token slice `&tokens[..len]`.
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.tokens[..self.len]
    }
}

/// Pure-integer verification receipt comparing drafted tokens against target verification tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VerificationReceipt {
    pub accepted_prefix: usize,
    pub total_drafted: usize,
    pub bonus_token: Option<u32>,
    pub acceptance_ratio: Ratio,
    pub mode: ExecutionMode,
}

/// Deterministic speculative decoding and prefix verification.
pub struct GravitationalLensing;

impl GravitationalLensing {
    /// Generate a deterministic speculative token draft of length `count.min(MAX_SPEC_DRAFT)`.
    pub fn speculate_timelines(seed_token: u32, count: usize) -> SpeculativeDraft {
        let len = count.min(MAX_SPEC_DRAFT);
        let mut tokens = [0u32; MAX_SPEC_DRAFT];
        let mut state = (seed_token as u64).wrapping_add(1);
        for slot in &mut tokens[..len] {
            state = splitmix64(state);
            *slot = (state & 0xFFFF) as u32;
        }
        SpeculativeDraft { tokens, len }
    }

    /// Verify a drafted token slice against target verifier outputs, returning exact prefix match
    /// length, optional bonus token, and exact rational acceptance ratio.
    pub fn verify_speculative_tokens(draft: &[u32], target: &[u32]) -> VerificationReceipt {
        let cmp_len = draft.len().min(target.len());
        let mut accepted_prefix = 0usize;
        while accepted_prefix < cmp_len && draft[accepted_prefix] == target[accepted_prefix] {
            accepted_prefix += 1;
        }
        let bonus_token = target.get(accepted_prefix).copied();
        let acceptance_ratio = if draft.is_empty() {
            Ratio::ZERO
        } else {
            Ratio::new(accepted_prefix as u64, draft.len() as u64)
        };
        VerificationReceipt {
            accepted_prefix,
            total_drafted: draft.len(),
            bonus_token,
            acceptance_ratio,
            mode: ExecutionMode::Measured,
        }
    }
}
