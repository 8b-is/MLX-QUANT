//! Bounded deterministic token ejection budget state machine.

use crate::ExecutionMode;

/// Snapshot of token generation progress against a hard step budget.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EjectionBudget {
    pub emitted_tokens: usize,
    pub max_tokens: usize,
    pub halted_on_eos: bool,
    pub mode: ExecutionMode,
}

/// Stateful deterministic token ejection port with hard budget and EOS termination.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WhiteHole {
    emitted_tokens: usize,
    max_tokens: usize,
    halted_on_eos: bool,
}

impl WhiteHole {
    /// Open a bounded token ejection port permitting up to `max_tokens` emissions.
    #[inline]
    pub const fn open_ejection_port(max_tokens: usize) -> Self {
        Self {
            emitted_tokens: 0,
            max_tokens,
            halted_on_eos: false,
        }
    }

    /// Attempt to emit `token`; returns `false` once `halted_on_eos` or `max_tokens` is reached.
    #[inline]
    pub fn emit_step(&mut self, token: u32, eos_token: u32) -> bool {
        if self.halted_on_eos || self.emitted_tokens >= self.max_tokens {
            return false;
        }
        self.emitted_tokens += 1;
        if token == eos_token {
            self.halted_on_eos = true;
        }
        true
    }

    /// Inspect current ejection budget telemetry.
    #[inline]
    pub const fn status(&self) -> EjectionBudget {
        EjectionBudget {
            emitted_tokens: self.emitted_tokens,
            max_tokens: self.max_tokens,
            halted_on_eos: self.halted_on_eos,
            mode: ExecutionMode::Measured,
        }
    }
}
