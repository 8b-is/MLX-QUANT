//! BeeHive zero-float core primitives (`#![no_std]`):
//! - Tequila trapping-free deadzone reactivation STE (`arXiv:2509.23809`)
//! - Engram fixed-array O(1) N-gram hash table with Q15.16 gated residual merge (`arXiv:2601.07372`)
//! - Manifold-Constrained Hyper-Connections (`mHC`) 4x4 Sinkhorn-Knopp on Birkhoff polytope
//! - Deterministic `d`-regular Ramanujan bipartite expander edge generator
//! - Insect Mushroom-Body Winner-Take-All + novelty habituation + honeybee quorum router

use crate::ratio::{Q16_ONE, Q16_SHIFT, gcd_binary_u64, q16_mul, splitmix64};

/// Configuration for Tequila trapping-free ternary QAT STE updates (`arXiv:2509.23809`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TequilaConfig {
    /// Deadzone half-width in Q15.16 (`|w| < delta_q16` triggers dynamic reactivation bias).
    pub delta_q16: i32,
    /// Learning rate in Q15.16 (`655` ~= `0.01`).
    pub lr_q16: i32,
    /// Dynamic bias reactivation gain `lambda` in Q15.16 (`66` ~= `10^-3`).
    pub lambda_q16: i32,
}

impl Default for TequilaConfig {
    #[inline]
    fn default() -> Self {
        Self {
            delta_q16: Q16_ONE >> 2,
            lr_q16: 655,
            lambda_q16: 66,
        }
    }
}

/// Pure Q15.16 / `i64` Tequila trapping-free STE step (`arXiv:2509.23809`).
///
/// - When `|w| >= delta_q16`: `eff_grad = grad_q16`.
/// - When deadzone `|w| < delta_q16`: `eff_grad = grad_q16 + ((lambda_q16 * w) >> 16)`.
/// - Updates `w_new = clamp(w - ((eff_grad * lr_q16) >> 16), i32::MIN, i32::MAX)`.
pub fn tequila_ste_step_q16(weights_q16: &mut [i32], grads_q16: &[i32], cfg: TequilaConfig) {
    assert_eq!(
        weights_q16.len(),
        grads_q16.len(),
        "weights_q16 and grads_q16 must match in length"
    );
    let delta = cfg.delta_q16 as i64;
    let lr = cfg.lr_q16 as i64;
    let lambda = cfg.lambda_q16 as i64;

    for (w_slot, &grad) in weights_q16.iter_mut().zip(grads_q16.iter()) {
        let w = *w_slot as i64;
        let abs_w = w.abs();
        let eff_grad = if abs_w < delta {
            let dyn_bias = (lambda * w) >> Q16_SHIFT;
            (grad as i64) + dyn_bias
        } else {
            grad as i64
        };
        let update = eff_grad.saturating_mul(lr) >> Q16_SHIFT;
        let next = w.saturating_sub(update);
        *w_slot = next.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

/// Zero-alloc fixed-array O(1) N-gram conditional memory table (`arXiv:2601.07372`)
/// storing Q15.16 vectors and fusing them via Q15.16 gated residual updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramTable<const SLOTS: usize, const DIM: usize> {
    pub entries: [[i32; DIM]; SLOTS],
}

impl<const SLOTS: usize, const DIM: usize> Default for EngramTable<SLOTS, DIM> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const SLOTS: usize, const DIM: usize> EngramTable<SLOTS, DIM> {
    /// Construct a zero-initialized conditional memory table.
    #[inline]
    pub const fn new() -> Self {
        Self {
            entries: [[0i32; DIM]; SLOTS],
        }
    }

    /// Number of hash slots in the table.
    #[inline]
    pub const fn capacity_slots(&self) -> usize {
        SLOTS
    }

    /// Embedding vector dimension.
    #[inline]
    pub const fn dim(&self) -> usize {
        DIM
    }

    /// Deterministic O(1) SplitMix64 N-gram hash mapped into `0..SLOTS`.
    #[inline]
    pub const fn hash_slot(ngram: &[u32], head_idx: usize) -> usize {
        if SLOTS == 0 {
            return 0;
        }
        let mut h = splitmix64(
            (head_idx as u64)
                .wrapping_add(1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let mut i = 0usize;
        while i < ngram.len() {
            h = splitmix64(h ^ (ngram[i] as u64));
            i += 1;
        }
        (h as usize) % SLOTS
    }

    /// Write `vec_q16` into `slot` (no-op if `slot >= SLOTS`).
    #[inline]
    pub fn insert(&mut self, slot: usize, vec_q16: &[i32; DIM]) {
        if slot < SLOTS {
            self.entries[slot] = *vec_q16;
        }
    }

    /// Hash `(ngram, head_idx)` and store `vec_q16`, returning the resolved slot index.
    #[inline]
    pub fn insert_ngram(&mut self, ngram: &[u32], head_idx: usize, vec_q16: &[i32; DIM]) -> usize {
        if SLOTS == 0 {
            return 0;
        }
        let slot = Self::hash_slot(ngram, head_idx);
        self.entries[slot] = *vec_q16;
        slot
    }

    /// Look up a slot directly.
    #[inline]
    pub const fn lookup(&self, slot: usize) -> Option<&[i32; DIM]> {
        if slot < SLOTS {
            Some(&self.entries[slot])
        } else {
            None
        }
    }

    /// Hash `(ngram, head_idx)` and return its stored embedding slice (or `None` if `SLOTS == 0`).
    #[inline]
    pub const fn lookup_ngram(&self, ngram: &[u32], head_idx: usize) -> Option<&[i32; DIM]> {
        if SLOTS == 0 {
            return None;
        }
        Some(&self.entries[Self::hash_slot(ngram, head_idx)])
    }

    /// Average multi-head retrieved N-gram embeddings into `mean_out` in `i64`.
    fn gather_mean_q16(&self, ngram: &[u32], num_heads: usize, mean_out: &mut [i32; DIM]) {
        if SLOTS == 0 || DIM == 0 {
            mean_out.fill(0);
            return;
        }
        let heads = if num_heads == 0 { 1 } else { num_heads };
        let mut acc = [0i64; DIM];
        for h in 0..heads {
            let slot = Self::hash_slot(ngram, h);
            let row = &self.entries[slot];
            for (dst, &val) in acc.iter_mut().zip(row.iter()) {
                *dst += val as i64;
            }
        }
        let div = heads as i64;
        for (dst, &sum) in mean_out.iter_mut().zip(acc.iter()) {
            *dst = (sum / div).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    /// Fuse retrieved N-gram memory into `hidden_q16` using an explicit `gate_q16` clamped to `[0, Q16_ONE]`.
    pub fn gated_merge_with_gate_q16(
        &self,
        hidden_q16: &mut [i32; DIM],
        ngram: &[u32],
        num_heads: usize,
        gate_q16: i32,
    ) {
        if SLOTS == 0 || DIM == 0 {
            return;
        }
        let mut mem_mean = [0i32; DIM];
        self.gather_mean_q16(ngram, num_heads, &mut mem_mean);
        let gate = gate_q16.clamp(0, Q16_ONE);
        for (h_val, &m_val) in hidden_q16.iter_mut().zip(mem_mean.iter()) {
            *h_val = h_val.saturating_add(q16_mul(gate, m_val));
        }
    }

    /// Context-aware Q15.16 gated residual merge:
    /// computes `dot(hidden_q16, mem_mean)` -> piecewise-linear Q15.16 gate in `[0, Q16_ONE]`,
    /// adds `gate * mem_mean` to `hidden_q16`, and returns the computed `gate_q16`.
    pub fn gated_residual_merge_q16(
        &self,
        hidden_q16: &mut [i32; DIM],
        ngram: &[u32],
        num_heads: usize,
    ) -> i32 {
        if SLOTS == 0 || DIM == 0 {
            return 0;
        }
        let mut mem_mean = [0i32; DIM];
        self.gather_mean_q16(ngram, num_heads, &mut mem_mean);

        let mut dot_q16: i64 = 0;
        for (&h_val, &m_val) in hidden_q16.iter().zip(mem_mean.iter()) {
            dot_q16 = dot_q16.saturating_add(((h_val as i64) * (m_val as i64)) >> Q16_SHIFT);
        }
        let mean_dot = dot_q16 / (DIM as i64);
        // Piecewise-linear integer sigmoid approximation centered at 0.5 * Q16_ONE.
        let gate = ((mean_dot >> 2) + ((Q16_ONE as i64) >> 1)).clamp(0, Q16_ONE as i64) as i32;
        for (h_val, &m_val) in hidden_q16.iter_mut().zip(mem_mean.iter()) {
            *h_val = h_val.saturating_add(q16_mul(gate, m_val));
        }
        gate
    }
}

/// Project a positive 4x4 matrix onto the Birkhoff polytope of doubly stochastic matrices
/// in pure Q15.16 integer arithmetic (`iters` Sinkhorn-Knopp row/column normalizations).
///
/// Entries `<= 0` are floored to `1` (`2^-16`) before normalization so every row and column
/// stays strictly positive and converges smoothly toward `Q16_ONE` (`65_536`).
pub fn mhc_sinkhorn_4x4_q16(raw_q16: [[i32; 4]; 4], iters: u32) -> [[i32; 4]; 4] {
    let mut m = [[0u64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let pos = if raw_q16[r][c] < 1 {
                1u64
            } else {
                raw_q16[r][c] as u64
            };
            m[r][c] = pos << Q16_SHIFT;
        }
    }

    for _ in 0..iters {
        for row in &mut m {
            let row_sum =
                (row[0] as u128) + (row[1] as u128) + (row[2] as u128) + (row[3] as u128);
            for cell in row.iter_mut() {
                let scaled = (((*cell as u128) << 32) + (row_sum >> 1)) / row_sum;
                *cell = (scaled as u64).max(1);
            }
        }
        for c in 0..4 {
            let col_sum =
                (m[0][c] as u128) + (m[1][c] as u128) + (m[2][c] as u128) + (m[3][c] as u128);
            for row in &mut m {
                let scaled = (((row[c] as u128) << 32) + (col_sum >> 1)) / col_sum;
                row[c] = (scaled as u64).max(1);
            }
        }
    }

    let mut out = [[0i32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let rounded = (m[r][c] + (1u64 << (Q16_SHIFT - 1))) >> Q16_SHIFT;
            out[r][c] = rounded.min(i32::MAX as u64) as i32;
        }
    }
    out
}

/// Generate a deterministic `d`-regular bipartite expander edge list `(row, col)`
/// (`d = degree.min(in_dim)`) into `edges_out` with zero heap allocations.
///
/// Guarantees every row `0..out_dim` connects to `d` distinct columns in `0..in_dim`
/// with balanced column degree across blocks. Returns total edges written (`out_dim * d`).
pub fn ramanujan_bipartite_edges(
    out_dim: usize,
    in_dim: usize,
    degree: usize,
    seed: u64,
    edges_out: &mut [(usize, usize)],
) -> usize {
    if out_dim == 0 || in_dim == 0 || degree == 0 {
        return 0;
    }
    let d = degree.min(in_dim);
    let total = out_dim
        .checked_mul(d)
        .expect("ramanujan_bipartite_edges size overflow");
    assert!(edges_out.len() >= total, "edges_out buffer too small");

    let in_u64 = in_dim as u64;
    let mut cached_block = usize::MAX;
    let mut block_a: u128 = 1;
    let mut block_b: u128 = 0;

    for r in 0..out_dim {
        let row_base = r * d;
        for k in 0..d {
            let edge_idx = row_base + k;
            let block = edge_idx / in_dim;
            let pos = edge_idx % in_dim;
            if block != cached_block {
                cached_block = block;
                let mut h = splitmix64(seed ^ ((block as u64).wrapping_add(1).wrapping_mul(0x517C_C1B7_2722_0A95)));
                let mut a = (h % in_u64).max(1);
                while in_u64 > 1 && gcd_binary_u64(a, in_u64) != 1 {
                    a += 1;
                    if a >= in_u64 {
                        a = 1;
                    }
                }
                h = splitmix64(h);
                block_a = a as u128;
                block_b = (h % in_u64) as u128;
            }
            let mut col = ((block_a * (pos as u128) + block_b) % (in_u64 as u128)) as usize;
            // Ensure intra-row uniqueness across the k already-chosen edges for row r.
            while edges_out[row_base..row_base + k]
                .iter()
                .any(|&(_, prev_col)| prev_col == col)
            {
                col += 1;
                if col >= in_dim {
                    col = 0;
                }
            }
            edges_out[row_base + k] = (r, col);
        }
    }
    total
}

/// Output of [`mushroom_body_wta_route`] carrying top-`K` selected experts and normalized Q15.16 weights.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WtaRouteResult<const K: usize> {
    pub indices: [usize; K],
    pub weights_q16: [i32; K],
    pub active_count: usize,
    pub quorum_met: bool,
}

/// Insect Mushroom-Body Winner-Take-All (`TopK(ReLU(scores) / (1 + habituation))`) +
/// honeybee quorum router in pure Q15.16 (`#![no_std]`, zero heap allocations).
///
/// Active weights sum to exact [`Q16_ONE`] whenever at least one input score is `> 0`.
pub fn mushroom_body_wta_route<const K: usize>(
    scores_q16: &[i32],
    habituation_q16: Option<&mut [i32]>,
    quorum_threshold_q16: i32,
) -> WtaRouteResult<K> {
    let mut result = WtaRouteResult {
        indices: [0usize; K],
        weights_q16: [0i32; K],
        active_count: 0,
        quorum_met: false,
    };
    if K == 0 || scores_q16.is_empty() {
        return result;
    }

    let mut top_scores = [0i64; K];
    let mut top_len = 0usize;

    for (idx, &raw) in scores_q16.iter().enumerate() {
        if raw <= 0 {
            continue;
        }
        let modulated = match habituation_q16.as_deref() {
            Some(hab) if idx < hab.len() => {
                let denom = (Q16_ONE as i64) + (hab[idx].max(0) as i64);
                ((raw as i64) << Q16_SHIFT) / denom
            }
            _ => raw as i64,
        };
        if modulated <= 0 {
            continue;
        }

        if top_len < K {
            let mut pos = top_len;
            while pos > 0 && modulated > top_scores[pos - 1] {
                top_scores[pos] = top_scores[pos - 1];
                result.indices[pos] = result.indices[pos - 1];
                pos -= 1;
            }
            top_scores[pos] = modulated;
            result.indices[pos] = idx;
            top_len += 1;
        } else if modulated > top_scores[K - 1] {
            let mut pos = K - 1;
            while pos > 0 && modulated > top_scores[pos - 1] {
                top_scores[pos] = top_scores[pos - 1];
                result.indices[pos] = result.indices[pos - 1];
                pos -= 1;
            }
            top_scores[pos] = modulated;
            result.indices[pos] = idx;
        }
    }

    if top_len == 0 {
        if let Some(hab) = habituation_q16 {
            for slot in hab.iter_mut() {
                *slot = q16_mul((*slot).max(0), 58_982);
            }
        }
        return result;
    }

    let total_top: i64 = top_scores[..top_len].iter().sum();
    if total_top <= 0 {
        return result;
    }

    let top_weight_q16 = ((top_scores[0] << Q16_SHIFT) / total_top) as i32;
    let quorum = quorum_threshold_q16.max(0);
    result.quorum_met = top_weight_q16 >= quorum;

    let mut active = 0usize;
    let mut kept_scores = [0i64; K];
    for j in 0..top_len {
        let w = ((top_scores[j] << Q16_SHIFT) / total_top) as i32;
        if j == 0 || w >= quorum {
            kept_scores[active] = top_scores[j];
            result.indices[active] = result.indices[j];
            active += 1;
        }
    }
    for j in active..K {
        result.indices[j] = 0;
        result.weights_q16[j] = 0;
    }

    // Re-normalize kept winners so active weights sum exactly to Q16_ONE.
    let active_score_sum: i64 = kept_scores[..active].iter().sum();
    if active_score_sum > 0 {
        let mut assigned_sum = 0i32;
        for j in 0..active {
            let w = ((kept_scores[j] << Q16_SHIFT) / active_score_sum) as i32;
            result.weights_q16[j] = w;
            assigned_sum += w;
        }
        result.weights_q16[0] += Q16_ONE - assigned_sum;
    }
    result.active_count = active;

    // Update decaying novelty habituation counters (0.9 decay + 0.1 firing bump).
    if let Some(hab) = habituation_q16 {
        for slot in hab.iter_mut() {
            *slot = q16_mul((*slot).max(0), 58_982);
        }
        for &winner_idx in &result.indices[..active] {
            if winner_idx < hab.len() {
                hab[winner_idx] = hab[winner_idx].saturating_add(6_554);
            }
        }
    }

    result
}
