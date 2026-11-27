//! Fixed-capacity, zero-heap circular Q15.16 Key-Value (KV) ring buffer.

use crate::ExecutionMode;
use mlx_quant_core::ratio::Q16_SHIFT;

/// Single Q15.16 Key-Value slot stored in [`CosmicMicrowaveBackground`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct KvEntryQ16 {
    pub key_q16: i32,
    pub value_q16: i32,
    pub token_pos: u32,
}

/// Descriptive window metadata emitted by [`CosmicMicrowaveBackground::permeate_context_window`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CmbWindowInfo {
    pub capacity_tokens: usize,
    pub bytes_per_slot: usize,
    pub mode: ExecutionMode,
}

/// Fixed-capacity circular Q15.16 KV ring buffer (`CAP` entries on stack/struct, zero `Vec`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CosmicMicrowaveBackground<const CAP: usize> {
    ring: [KvEntryQ16; CAP],
    head: usize,
    len: usize,
    total_written: u64,
}

impl<const CAP: usize> Default for CosmicMicrowaveBackground<CAP> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> CosmicMicrowaveBackground<CAP> {
    /// Construct an empty circular Q15.16 KV cache.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ring: [KvEntryQ16 {
                key_q16: 0,
                value_q16: 0,
                token_pos: 0,
            }; CAP],
            head: 0,
            len: 0,
            total_written: 0,
        }
    }

    /// Maximum ring capacity in KV slots.
    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    /// Number of valid entries currently retained (`<= CAP`).
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no entries are stored.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of pushes since construction or last [`Self::clear`].
    #[inline]
    pub const fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Push a Q15.16 `(key_q16, value_q16, token_pos)` entry, overwriting the oldest entry when full.
    #[inline]
    pub fn push(&mut self, key_q16: i32, value_q16: i32, token_pos: u32) {
        if CAP == 0 {
            return;
        }
        self.ring[self.head] = KvEntryQ16 {
            key_q16,
            value_q16,
            token_pos,
        };
        self.head = (self.head + 1) % CAP;
        if self.len < CAP {
            self.len += 1;
        }
        self.total_written = self.total_written.wrapping_add(1);
    }

    /// Retrieve an active entry by logical FIFO order (`0` = oldest retained entry).
    #[inline]
    pub fn get_oldest(&self, idx: usize) -> Option<KvEntryQ16> {
        if idx >= self.len || CAP == 0 {
            return None;
        }
        let start = (self.head + CAP - self.len) % CAP;
        Some(self.ring[(start + idx) % CAP])
    }

    /// Retrieve an active entry relative to the newest push (`0` = most recent entry).
    #[inline]
    pub fn get_latest(&self, back: usize) -> Option<KvEntryQ16> {
        if back >= self.len || CAP == 0 {
            return None;
        }
        let idx = (self.head + CAP - 1 - back) % CAP;
        Some(self.ring[idx])
    }

    /// Clear all stored KV entries.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Pure-integer Q15.16 sum of `query_q16 * key_q16 >> 16` across all retained entries.
    pub fn dot_keys_q16(&self, query_q16: i32) -> i64 {
        let mut acc = 0i64;
        for idx in 0..self.len {
            if let Some(entry) = self.get_oldest(idx) {
                let prod = ((query_q16 as i64) * (entry.key_q16 as i64)) >> Q16_SHIFT;
                acc = acc.saturating_add(prod);
            }
        }
        acc
    }

    /// Compute context-window metadata descriptor without heap allocation.
    #[inline]
    pub const fn permeate_context_window(context_size: usize) -> CmbWindowInfo {
        CmbWindowInfo {
            capacity_tokens: context_size,
            bytes_per_slot: core::mem::size_of::<KvEntryQ16>(),
            mode: ExecutionMode::Measured,
        }
    }
}
