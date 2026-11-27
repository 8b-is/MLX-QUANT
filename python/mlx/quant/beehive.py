"""
BeeHive-ERNIE v2.2.0 Architecture Modules for MLX-QUANT
======================================================
Integrates graph theory, insect mushroom-body/honeybee neurobiology,
ERNIE-5.5-Flash sparse/latent primitives, and Dual Outer RL (`RL` + `HVE-RL`):

  - Manifold-Constrained Hyper-Connections (`mHC`, arXiv:2512.24880)
  - Ramanujan Bipartite Expander Sparsity & Spectral Gap Report (`SPARSE`)
  - Latent Compressed Attention (`MLA` + `HCA` + `VHA`, ERNIE-5.5-Flash, `LATENT`)
  - MushroomBodyRouter (Kenyon-cell sparse expansion, WTA, habituation & quorum)
  - MultiMaxOutputHead (ERNIE-5.5-Flash multi-branch output head)
  - HveRlController (Tactical Dr. GRPO RL + Strategic `{logs, tool_calls, reasoning}` HVE-RL)
"""

import json
import math
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .rational import RATIO_ZERO, Ratio


# ---------------------------------------------------------------------------
# 1. Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880)
# ---------------------------------------------------------------------------


def sinkhorn_knopp_mx(logits: mx.array, iters: int = 6) -> mx.array:
    """
    Project a square logit matrix `[S, S]` onto the Birkhoff polytope of doubly
    stochastic matrices via iterative Sinkhorn-Knopp normalization.
    """
    shifted = logits - mx.max(logits)
    mat = mx.exp(shifted)
    for _ in range(max(1, iters)):
        mat = mat / mx.maximum(mx.sum(mat, axis=-1, keepdims=True), 1e-8)
        mat = mat / mx.maximum(mx.sum(mat, axis=-2, keepdims=True), 1e-8)
    return mat


class ManifoldHyperConnection(nn.Module):
    """
    4-stream parallel residual honeycomb constrained to the Birkhoff polytope (`mHC`).
    Conserves global signal intensity across deep ternary/sparse blocks.
    """

    def __init__(
        self,
        d_model: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 6,
    ) -> None:
        super().__init__()
        if num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        self.d_model = d_model
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Initialize near identity matrix so step-0 residual flow stays exact
        self.mix_logits = mx.eye(num_streams, dtype=mx.float32) * 3.0
        self.pre_logits = mx.zeros((num_streams,), dtype=mx.float32)
        self.post_logits = mx.zeros((num_streams,), dtype=mx.float32)

    def read_stream(self, streams: mx.array) -> mx.array:
        """Collapse `[B, T, S, D]` streams into a single `[B, T, D]` layer input."""
        w_pre = mx.softmax(self.pre_logits, axis=0)
        return mx.sum(streams * w_pre.reshape(1, 1, self.num_streams, 1), axis=-2)

    def write_and_mix(self, streams: mx.array, delta: mx.array) -> mx.array:
        """Mix `[B, T, S, D]` on the Birkhoff polytope and distribute `delta` `[B, T, D]`."""
        h_res = sinkhorn_knopp_mx(self.mix_logits, iters=self.sinkhorn_iters)
        # [B, T, D, S] @ [S, S]^T -> [B, T, D, S] -> [B, T, S, D]
        s_t = streams.transpose(0, 1, 3, 2)
        mixed = mx.matmul(s_t, h_res.T).transpose(0, 1, 3, 2)

        w_post = mx.sigmoid(self.post_logits) * 2.0
        update = mx.expand_dims(delta, axis=-2) * w_post.reshape(1, 1, self.num_streams, 1)
        return mixed + update

    def __call__(self, streams: mx.array, block_fn) -> mx.array:
        x_in = self.read_stream(streams)
        delta = block_fn(x_in)
        return self.write_and_mix(streams, delta)


# ---------------------------------------------------------------------------
# 2. Ramanujan Bipartite Expander Mask & Spectral Gap (`SPARSE`)
# ---------------------------------------------------------------------------


def ramanujan_bipartite_mask(
    out_dim: int,
    in_dim: int,
    degree: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Construct a deterministic `degree`-row-regular bipartite expander mask `{0, 1}`
    of shape `[out_dim, in_dim]` with balanced column degrees.
    """
    if out_dim <= 0 or in_dim <= 0 or degree <= 0:
        raise ValueError("out_dim, in_dim, and degree must be positive")
    d = min(degree, in_dim)
    rng = np.random.default_rng(seed)
    mask = np.zeros((out_dim, in_dim), dtype=np.float32)

    total_edges = out_dim * d
    blocks = (total_edges + in_dim - 1) // in_dim
    pool = np.concatenate([rng.permutation(in_dim) for _ in range(blocks)])[:total_edges]
    col_loads = np.zeros((in_dim,), dtype=np.int32)

    for r in range(out_dim):
        candidates = pool[r * d : (r + 1) * d]
        chosen = []
        seen = set()
        for c in candidates.tolist():
            if c not in seen:
                chosen.append(c)
                seen.add(c)
        # Resolve any rare intra-row collision using least-loaded available columns
        if len(chosen) < d:
            avail = [c for c in np.argsort(col_loads).tolist() if c not in seen]
            for c in avail[: d - len(chosen)]:
                chosen.append(c)
                seen.add(c)
        for c in chosen:
            mask[r, c] = 1.0
            col_loads[c] += 1

    return mask


def spectral_gap_report(mask: np.ndarray) -> Dict[str, float]:
    """Compute bipartite SVD eigenvalues `lambda_1`, `lambda_2`, spectral gap, and Alon-Boppana ref."""
    sv = np.linalg.svd(mask.astype(np.float64), compute_uv=False)
    lambda_1 = float(sv[0]) if sv.size >= 1 else 0.0
    lambda_2 = float(sv[1]) if sv.size >= 2 else 0.0
    d_avg = float(mask.sum(axis=1).mean()) if mask.ndim == 2 and mask.shape[0] > 0 else 0.0
    alon_boppana = 2.0 * math.sqrt(max(d_avg - 1.0, 0.0))
    return {
        "out_dim": float(mask.shape[0]),
        "in_dim": float(mask.shape[1]),
        "row_degree": d_avg,
        "density": float(mask.mean()),
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "spectral_gap": lambda_1 - lambda_2,
        "alon_boppana_ref": alon_boppana,
        "is_expander": bool(lambda_1 > lambda_2),
    }


# ---------------------------------------------------------------------------
# 3. ERNIE-5.5-Flash Latent Compressed Attention (`MLA` + `HCA` + `VHA`)
# ---------------------------------------------------------------------------


class LatentCompressedAttention(nn.Module):
    """
    Multi-head Latent Attention (`kv_lora_rank`) + Heavily Compressed Attention
    (`hca_stride`) + Virtual Head Attention gate (`vha_gate`).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int = 64,
        hca_stride: int = 16,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.hca_stride = max(1, hca_stride)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=1e-6)
        self.k_up_proj = nn.Linear(kv_lora_rank, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, d_model, bias=False)
        self.vha_gate = nn.Linear(d_model, n_heads, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = nn.RoPE(self.head_dim)

    def kv_cache_compression_ratio(self) -> float:
        """Ratio of full standard `[K, V]` cache (`2 * d_model`) to latent `kv_lora_rank`."""
        return (2.0 * self.d_model) / max(1, self.kv_lora_rank)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Latent KV bottleneck (`LATENT`)
        c_kv = self.kv_norm(self.kv_down_proj(x))
        k = self.k_up_proj(c_kv).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_up_proj(c_kv).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q_rope = self.rope(q)
        k_rope = self.rope(k)

        fine_attn = mx.fast.scaled_dot_product_attention(
            q_rope, k_rope, v, scale=self.head_dim ** -0.5, mask="causal"
        )

        # Heavily Compressed Attention (HCA) landmark stream over completed stride blocks
        if self.hca_stride > 1 and seq_len >= 2 * self.hca_stride:
            n_chunks = seq_len // self.hca_stride
            usable = n_chunks * self.hca_stride
            c_blocks = c_kv[:, :usable, :].reshape(bsz, n_chunks, self.hca_stride, self.kv_lora_rank)
            c_summary = mx.mean(c_blocks, axis=2)  # [B, n_chunks, kv_lora_rank]

            k_hca = (
                self.k_up_proj(c_summary)
                .reshape(bsz, n_chunks, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v_hca = (
                self.v_up_proj(c_summary)
                .reshape(bsz, n_chunks, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

            # Strict causal mask: token t may attend to chunk m iff chunk ends strictly before t
            q_pos = mx.arange(seq_len).reshape(1, 1, seq_len, 1)
            chunk_end = ((mx.arange(n_chunks) + 1) * self.hca_stride - 1).reshape(1, 1, 1, n_chunks)
            allowed = chunk_end < q_pos
            # Also guarantee at least chunk 0 is unmasked or gracefully zero-masked via -1e4
            hca_mask = mx.where(allowed, 0.0, -1e4).astype(q.dtype)
            # If no past chunk exists (early tokens t < hca_stride), VHA blend still stays finite
            hca_attn = mx.fast.scaled_dot_product_attention(
                q, k_hca, v_hca, scale=self.head_dim ** -0.5, mask=hca_mask
            )
            # Zero out HCA contribution for tokens before the first completed chunk
            has_past = (q_pos > (self.hca_stride - 1)).astype(q.dtype)
            gate = mx.sigmoid(self.vha_gate(x)).transpose(0, 2, 1).reshape(bsz, self.n_heads, seq_len, 1)
            blended = fine_attn + has_past * gate * (hca_attn - fine_attn)
        else:
            blended = fine_attn

        merged = blended.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)
        return self.o_proj(merged)


# ---------------------------------------------------------------------------
# 4. Biological Mushroom-Body + Honeybee Quorum Router (`Ternary workers`)
# ---------------------------------------------------------------------------


class MushroomBodyRouter(nn.Module):
    """
    Insect Mushroom-Body Kenyon-cell sparse router with novelty habituation
    (*The Fly That Stopped*, Sep 2026) and honeybee quorum thresholding.
    """

    def __init__(
        self,
        d_model: int,
        n_kenyon: int = 128,
        kc_degree: int = 8,
        kc_top_k: int = 16,
        num_workers: int = 4,
        top_k_workers: int = 2,
        quorum_threshold: float = 0.08,
        habituation_gain: float = 0.5,
        habituation_decay: float = 0.9,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_kenyon = n_kenyon
        self.kc_top_k = min(kc_top_k, n_kenyon)
        self.num_workers = num_workers
        self.top_k_workers = min(top_k_workers, num_workers)
        self.quorum_threshold = quorum_threshold
        self.habituation_gain = habituation_gain
        self.habituation_decay = habituation_decay

        mask_np = ramanujan_bipartite_mask(n_kenyon, d_model, degree=kc_degree, seed=seed)
        self._kc_mask = mx.array(mask_np, dtype=mx.float32)
        self.pn_to_kc = nn.Linear(d_model, n_kenyon, bias=False)
        self.mbon_proj = nn.Linear(n_kenyon, num_workers, bias=True)
        self._habituation = mx.zeros((n_kenyon,), dtype=mx.float32)

    def _sparse_wta_kenyon(self, h: mx.array) -> mx.array:
        """Project through Ramanujan PN->KC mask and keep top-k Kenyon cells per token."""
        w_sparse = self.pn_to_kc.weight * mx.stop_gradient(self._kc_mask)
        raw_kc = nn.relu(h @ w_sparse.T)
        if self.kc_top_k >= self.n_kenyon:
            return raw_kc
        # Threshold at the k-th largest value per token for differentiable WTA masking
        sorted_vals = mx.sort(raw_kc, axis=-1)
        kth_val = sorted_vals[..., self.n_kenyon - self.kc_top_k : self.n_kenyon - self.kc_top_k + 1]
        wta_mask = mx.stop_gradient((raw_kc >= kth_val).astype(raw_kc.dtype))
        return raw_kc * wta_mask

    def update_habituation(self, kc_code: mx.array) -> None:
        """Update decaying Kenyon-cell firing counter (reward-free novelty prior)."""
        mean_fire = mx.mean((mx.stop_gradient(kc_code) > 0.0).astype(mx.float32), axis=(0, 1))
        self._habituation = (
            self.habituation_decay * mx.stop_gradient(self._habituation)
            + (1.0 - self.habituation_decay) * mean_fire
        )

    def __call__(
        self, hidden: mx.array, update_state: bool = True
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        kc_code = self._sparse_wta_kenyon(hidden)
        novelty_mod = 1.0 / (1.0 + self.habituation_gain * mx.stop_gradient(self._habituation))
        kc_modulated = kc_code * novelty_mod

        if update_state:
            self.update_habituation(kc_code)

        worker_logits = self.mbon_proj(kc_modulated)
        probs = mx.softmax(worker_logits, axis=-1)

        # Honeybee quorum gate + top-k worker dispatch
        sorted_w = mx.sort(probs, axis=-1)
        kth_w = sorted_w[..., self.num_workers - self.top_k_workers : self.num_workers - self.top_k_workers + 1]
        active = (probs >= kth_w) & (probs >= self.quorum_threshold)
        # Ensure at least the top-1 worker stays active if all fall below quorum
        max_w = mx.max(probs, axis=-1, keepdims=True)
        active = active | (probs >= max_w)
        gated = probs * mx.stop_gradient(active.astype(probs.dtype))
        weights = gated / mx.maximum(mx.sum(gated, axis=-1, keepdims=True), 1e-8)

        kc_density = mx.mean((kc_code > 0.0).astype(mx.float32))
        metrics = {
            "kc_active_density": kc_density,
            "mean_habituation": mx.mean(self._habituation),
            "worker_entropy": -mx.mean(mx.sum(probs * mx.log(mx.maximum(probs, 1e-8)), axis=-1)),
        }
        return weights, metrics


# ---------------------------------------------------------------------------
# 5. ERNIE-5.5-Flash MultiMax Output Head
# ---------------------------------------------------------------------------


class MultiMaxOutputHead(nn.Module):
    """
    ERNIE-5.5-Flash MultiMax output head combining `num_branches` linear projections
    with temperature-scaled branch gating to reduce softmax bottlenecks.
    """

    def __init__(self, d_model: int, vocab_size: int, num_branches: int = 2) -> None:
        super().__init__()
        self.num_branches = max(1, num_branches)
        self.branches = [nn.Linear(d_model, vocab_size, bias=False) for _ in range(self.num_branches)]
        self.branch_gate = nn.Linear(d_model, self.num_branches, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        gate = mx.softmax(self.branch_gate(x), axis=-1)  # [B, T, K]
        logits_acc = None
        for idx, head in enumerate(self.branches):
            b_logits = head(x)
            w = gate[..., idx : idx + 1]
            term = b_logits * w
            logits_acc = term if logits_acc is None else (logits_acc + term)
        return logits_acc


# ---------------------------------------------------------------------------
# 6. Dual Outer RL Loops (`Tactical RL` + `Strategic HVE-RL`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HveRlReport:
    group_has_signal: bool
    tactical_advantages: Tuple[float, ...]
    raw_trace_bytes: int
    compressed_trace_bytes: int
    trace_compression_ratio: Ratio
    verified_pass: bool
    canary_recall_ratio: Ratio
    hve_rl_reward: Ratio


class HveRlController:
    """
    Coordinates the two outer RL loops in the BeeHive sketch:
      1. Tactical `RL`: Dr. GRPO unbiased group advantage + DAPO variance filter.
      2. Strategic `HVE-RL`: MDL compression of `{logs, tool_calls, reasoning}`
         gated by deterministic execution verification and Linger-and-Lose recall.
    """

    @staticmethod
    def compute_tactical_rl_advantages(rewards: Sequence[float]) -> Tuple[bool, list[float]]:
        if not rewards:
            return False, []
        has_signal = (max(rewards) - min(rewards)) > 1e-9
        if not has_signal:
            return False, [0.0] * len(rewards)
        mean_r = sum(rewards) / len(rewards)
        return True, [r - mean_r for r in rewards]

    @staticmethod
    def compress_trace_bundle(
        logs: str,
        tool_calls: Sequence[str],
        reasoning: str,
    ) -> Tuple[int, int, bytes]:
        payload = json.dumps(
            {
                "logs": logs,
                "tool_calls": list(tool_calls),
                "reasoning": reasoning,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        compressed = zlib.compress(payload, level=9)
        return len(payload), len(compressed), compressed

    @staticmethod
    def compute_hve_rl_reward(
        raw_bytes: int,
        compressed_bytes: int,
        verified_pass: bool,
        canary_recall_ratio: Ratio,
    ) -> Ratio:
        if not verified_pass or raw_bytes <= 0:
            return RATIO_ZERO
        comp = Ratio(raw_bytes, max(1, compressed_bytes))
        return comp.mul_exact(canary_recall_ratio)

    def evaluate_episode(
        self,
        rollout_rewards: Sequence[float],
        logs: str,
        tool_calls: Sequence[str],
        reasoning: str,
        verified_pass: bool,
        canary_recall_ratio: Ratio,
    ) -> HveRlReport:
        has_signal, advs = self.compute_tactical_rl_advantages(rollout_rewards)
        raw_b, comp_b, _ = self.compress_trace_bundle(logs, tool_calls, reasoning)
        comp_ratio = Ratio(raw_b, max(1, comp_b))
        hve_reward = self.compute_hve_rl_reward(
            raw_b, comp_b, verified_pass=verified_pass, canary_recall_ratio=canary_recall_ratio
        )
        return HveRlReport(
            group_has_signal=has_signal,
            tactical_advantages=tuple(round(a, 6) for a in advs),
            raw_trace_bytes=raw_b,
            compressed_trace_bytes=comp_b,
            trace_compression_ratio=comp_ratio,
            verified_pass=verified_pass,
            canary_recall_ratio=canary_recall_ratio,
            hve_rl_reward=hve_reward,
        )
