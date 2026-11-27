//! Pure-integer micro-batch scheduling and chunk execution over token slices.

use crate::ExecutionMode;

/// Deterministic micro-batch partition schedule.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MicroBatchSchedule {
    pub total_tokens: usize,
    pub micro_batch_size: usize,
    pub num_batches: usize,
    pub remainder: usize,
    pub mode: ExecutionMode,
}

/// Deterministic integer micro-batch planner and runner.
pub struct TimeDilation;

impl TimeDilation {
    /// Compute exact batch counts and remainder for `total_tokens` split by `micro_batch_size`.
    #[inline]
    pub const fn plan_micro_batches(
        total_tokens: usize,
        micro_batch_size: usize,
    ) -> MicroBatchSchedule {
        let eff_batch = if micro_batch_size == 0 {
            1
        } else {
            micro_batch_size
        };
        let num_batches = total_tokens.div_ceil(eff_batch);
        let remainder = if total_tokens == 0 {
            0
        } else {
            let rem = total_tokens % eff_batch;
            if rem == 0 { eff_batch } else { rem }
        };
        MicroBatchSchedule {
            total_tokens,
            micro_batch_size: eff_batch,
            num_batches,
            remainder,
            mode: ExecutionMode::Measured,
        }
    }

    /// Execute `f` sequentially across every micro-batch chunk of `tokens`, returning batch count.
    pub fn micro_batch_loop<T, F>(tokens: &[T], micro_batch_size: usize, mut f: F) -> usize
    where
        F: FnMut(usize, &[T]),
    {
        let eff_batch = micro_batch_size.max(1);
        let mut count = 0usize;
        for (idx, chunk) in tokens.chunks(eff_batch).enumerate() {
            f(idx, chunk);
            count += 1;
        }
        count
    }
}
