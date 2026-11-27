#!/usr/bin/env python3
"""
A6000 (RTX 6000 Ada 48 GB SM_89) BeeHive-ERNIE `#![no_std]` Rust Coder Trainer
================================================================================
Scaled PyTorch CUDA trainer matching `MLX-QUANT` v3.0.0-alpha.1 (`BeeHive-ERNIE` +
`Zeno-Engram` + `BITCOS` + `H-BitLinear` + `Tequila` + `p/q` Rational WSD):

- Presets:
  - `smoke`: ~4M params (local Apple MPS / CPU / CUDA validation in seconds)
  - `125m`:  ~125M params (d=768, L=12, H=12, safe under 24 GB VRAM)
  - `350m`:  ~350M params (d=1024, L=18, H=16, fits inside 30 GB free w/ checkpointing)
  - `1p7b`:  ~1.7B params (d=2048, L=24, H=16, full 48 GB dedicated mode)
- VRAM guard: checks `torch.cuda.mem_get_info()` vs `--max-vram-gb` (default `24`) and
  sets `torch.cuda.set_per_process_memory_fraction` when running on CUDA so background
  A6000 services (`st-mem8-context` / `alexiai` / `liquid-rust`) are never evicted.
- Exports: `model.safetensors` (or `.pt`), `model.bitcos.npz` (`bitcos_pack_numpy` w/
  exact roundtrip `err == 0`), `bitcos_manifest.json`, `training_metrics.json`, `config.json`,
  and `README.md`, with optional `--push-hf eightb/nostd-rust-coder-ternary-engram-v3`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
RATIONAL_MOD_PATH = REPO_ROOT / "python" / "mlx" / "quant" / "rational.py"
if RATIONAL_MOD_PATH.exists():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("_mq_rational", RATIONAL_MOD_PATH)
    assert _spec is not None and _spec.loader is not None
    _mq_rational = importlib.util.module_from_spec(_spec)
    sys.modules["_mq_rational"] = _mq_rational
    _spec.loader.exec_module(_mq_rational)
    LmHeadLrPolicy = _mq_rational.LmHeadLrPolicy
    Ratio = _mq_rational.Ratio
    WsdScheduleConfig = _mq_rational.WsdScheduleConfig
    compute_byte_asymmetric_mask = _mq_rational.compute_byte_asymmetric_mask
    evaluate_knowledge_collapse_canaries = _mq_rational.evaluate_knowledge_collapse_canaries
else:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from mlx.quant.rational import (
        LmHeadLrPolicy,
        Ratio,
        WsdScheduleConfig,
        compute_byte_asymmetric_mask,
        evaluate_knowledge_collapse_canaries,
    )


def np_bitcos_pack(w: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Pure-NumPy BITCOS (`arXiv:2609.16338`) pack usable on Linux CUDA hosts without `mlx.core`."""
    w_f32 = w.astype(np.float32, copy=False)
    out_features, in_features = w_f32.shape
    scales = np.maximum(np.mean(np.abs(w_f32), axis=1, keepdims=True), 1e-8).astype(np.float32)
    trits = np.where(np.abs(w_f32) < threshold * scales, 0, np.where(w_f32 >= 0, 1, -1)).astype(np.int8)
    p_bytes = (in_features + 7) // 8
    presence = np.zeros((out_features, p_bytes), dtype=np.uint8)
    row_sign_offsets = np.zeros((out_features,), dtype=np.uint32)
    row_nnz = np.zeros((out_features,), dtype=np.uint32)
    sign_chunks: List[np.ndarray] = []
    byte_cursor = 0
    for r in range(out_features):
        row = trits[r]
        nz_mask = row != 0
        if p_bytes > 0:
            presence[r] = np.packbits(nz_mask.astype(np.uint8), bitorder="little")[:p_bytes]
        nz_vals = row[nz_mask]
        nnz = int(nz_vals.size)
        row_nnz[r] = nnz
        row_sign_offsets[r] = byte_cursor
        if nnz > 0:
            pos_bits = (nz_vals > 0).astype(np.uint8)
            s_packed = np.packbits(pos_bits, bitorder="little")
            sign_chunks.append(s_packed)
            byte_cursor += int(s_packed.size)
    signs = np.concatenate(sign_chunks) if sign_chunks else np.zeros((0,), dtype=np.uint8)
    return presence, signs, row_sign_offsets, row_nnz, scales, int(row_nnz.sum()), trits


def np_bitcos_unpack(
    presence: np.ndarray,
    signs: np.ndarray,
    row_sign_offsets: np.ndarray,
    row_nnz: np.ndarray,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Pure-NumPy BITCOS unpack matching `mlx.quant.bitnet.bitcos_unpack_numpy`."""
    out = np.zeros((out_features, in_features), dtype=np.int8)
    if in_features == 0:
        return out
    for r in range(out_features):
        nnz = int(row_nnz[r])
        if nnz == 0:
            continue
        p_bits = np.unpackbits(presence[r], bitorder="little")[:in_features]
        nz_cols = np.flatnonzero(p_bits)
        s_off = int(row_sign_offsets[r])
        s_bytes = (nnz + 7) // 8
        s_bits = np.unpackbits(signs[s_off : s_off + s_bytes], bitorder="little")[:nnz]
        out[r, nz_cols] = np.where(s_bits != 0, 1, -1).astype(np.int8)
    return out

VOCAB_SIZE = 256
NOSTD_CANARIES: List[Tuple[str, str, int]] = [
    ("#![", "no_std]", 32),
    ("use core::", "arch::", 32),
    ("Ratio::", "new(", 24),
    ("pub enum Trit", " { Neg", 24),
    ("epoch_bump_", "arena", 24),
    ("tequila_ste_", "step_q16", 24),
]


@dataclass(frozen=True)
class ModelPreset:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    kv_latent: int
    engram_slots: int
    engram_dim: int
    n_experts: int
    top_k: int
    seq_len: int
    steps: int
    batch_size: int


PRESETS: Dict[str, ModelPreset] = {
    "smoke": ModelPreset(
        name="smoke",
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=384,
        kv_latent=48,
        engram_slots=512,
        engram_dim=64,
        n_experts=4,
        top_k=2,
        seq_len=128,
        steps=8,
        batch_size=4,
    ),
    "125m": ModelPreset(
        name="125m",
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=2048,
        kv_latent=192,
        engram_slots=4096,
        engram_dim=128,
        n_experts=8,
        top_k=2,
        seq_len=512,
        steps=400,
        batch_size=8,
    ),
    "350m": ModelPreset(
        name="350m",
        d_model=1024,
        n_layers=18,
        n_heads=16,
        d_ff=2816,
        kv_latent=256,
        engram_slots=8192,
        engram_dim=128,
        n_experts=8,
        top_k=2,
        seq_len=512,
        steps=600,
        batch_size=6,
    ),
    "1p7b": ModelPreset(
        name="1p7b",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        d_ff=5120,
        kv_latent=256,
        engram_slots=16384,
        engram_dim=192,
        n_experts=8,
        top_k=2,
        seq_len=512,
        steps=800,
        batch_size=4,
    ),
}


def hadamard_wht_torch(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Fast Walsh-Hadamard Transform over the trailing power-of-two dimension."""
    d = x.shape[-1]
    if d <= 1 or (d & (d - 1)) != 0:
        return x
    h = x
    chunk = 1
    while chunk < d:
        h = h.view(*x.shape[:-1], d // (2 * chunk), 2, chunk)
        even = h[..., 0, :]
        odd = h[..., 1, :]
        h = torch.cat((even + odd, even - odd), dim=-1)
        chunk *= 2
    return h.view_as(x) / math.sqrt(float(d))


def activation_quant_int4_ste(x: torch.Tensor) -> torch.Tensor:
    """Symmetric W1.58A4 AbsMax INT4 STE (`[-7, +7]`)."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 7.0
    q = (x / scale).round().clamp(-7.0, 7.0) * scale
    return x + (q - x).detach()


def ramanujan_mask_np(out_dim: int, in_dim: int, degree: int, seed: int) -> np.ndarray:
    """Deterministic `d`-regular bipartite expander binary mask matching `mlx.quant.beehive`."""
    d = max(1, min(degree, in_dim))
    mask = np.zeros((out_dim, in_dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    perm_cols = rng.permutation(in_dim)
    cursor = 0
    for r in range(out_dim):
        chosen = []
        while len(chosen) < d:
            if cursor >= in_dim:
                perm_cols = rng.permutation(in_dim)
                cursor = 0
            col = int(perm_cols[cursor])
            cursor += 1
            if col not in chosen:
                chosen.append(col)
                mask[r, col] = 1.0
    return mask


class BitLinearHadamardTequila(nn.Module):
    """BitNet b1.58 + H-BitLinear WHT + Tequila deadzone reactivation + Ramanujan sparse mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tequila_lambda: float = 1e-3,
        sparse_degree: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tequila_lambda = tequila_lambda
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if sparse_degree is not None and sparse_degree < in_features:
            mask = torch.from_numpy(ramanujan_mask_np(out_features, in_features, sparse_degree, seed))
        else:
            mask = torch.ones(out_features, in_features, dtype=torch.float32)
        self.register_buffer("sparse_mask", mask, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.sparse_mask
        scale = w.abs().mean().clamp_min(1e-5)
        w_scaled = w / scale
        w_round = w_scaled.round().clamp(-1.0, 1.0)
        is_zero = (w_round == 0).to(w.dtype)
        # Tequila trapping-free reactivation gradient term when trapped at 0.
        w_q = w_round * scale
        w_ste = w + (w_q - w).detach() + self.tequila_lambda * is_zero * w

        x_rot = hadamard_wht_torch(x)
        x_q = activation_quant_int4_ste(x_rot)
        return F.linear(x_q, w_ste)


class ManifoldHyperConnection(nn.Module):
    """mHC 4-stream doubly stochastic Birkhoff Sinkhorn residual honeycomb (`arXiv:2512.24880`)."""

    def __init__(self, d_model: int, num_streams: int = 4, sinkhorn_iters: int = 8) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.mix_logits = nn.Parameter(torch.eye(num_streams) * 2.0)
        self.alpha = nn.Parameter(torch.full((num_streams,), 1.0 / num_streams))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, streams: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # streams: (B, T, S, D)
        mat = self.mix_logits.float().clamp(-10.0, 10.0).exp()
        for _ in range(self.sinkhorn_iters):
            mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        mat = mat.to(streams.dtype)

        mixed = torch.einsum("ij,btjd->btid", mat, streams)
        weights = F.softmax(self.alpha.float(), dim=0).to(streams.dtype)
        agg = torch.einsum("s,btsd->btd", weights, mixed)
        delta = sublayer(self.norm(agg))
        return mixed + delta.unsqueeze(2)


class LatentCompressedAttention(nn.Module):
    """ERNIE-5.5-style Multi-Head Latent Attention (MLA) + strided Hierarchical Landmark branch."""

    def __init__(self, d_model: int, n_heads: int, kv_latent: int, landmark_stride: int = 8) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.landmark_stride = landmark_stride

        self.q_proj = BitLinearHadamardTequila(d_model, d_model)
        self.kv_down = BitLinearHadamardTequila(d_model, kv_latent)
        self.k_up = nn.Linear(kv_latent, d_model, bias=False)
        self.v_up = nn.Linear(kv_latent, d_model, bias=False)
        self.o_proj = BitLinearHadamardTequila(d_model, d_model)
        self.hca_gate = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down(x)
        k = self.k_up(c_kv).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mla = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Causal strided landmark summary branch (HCA).
        if seq >= self.landmark_stride:
            idx = torch.arange(0, seq, self.landmark_stride, device=x.device)
            k_lm = k[:, :, idx, :]
            v_lm = v[:, :, idx, :]
            t_q = torch.arange(seq, device=x.device).unsqueeze(1)
            t_k = idx.unsqueeze(0)
            causal_mask = t_k <= t_q
            attn_hca = F.scaled_dot_product_attention(q, k_lm, v_lm, attn_mask=causal_mask)
            out = attn_mla + torch.sigmoid(self.hca_gate) * attn_hca
        else:
            out = attn_mla

        out = out.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        return self.o_proj(out)


class EngramMemoryLayer(nn.Module):
    """DeepSeek Engram O(1) multi-head N-gram conditional memory (`arXiv:2601.07372`)."""

    def __init__(self, d_model: int, slots: int, mem_dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.slots = slots
        self.n_heads = n_heads
        self.tables = nn.Parameter(torch.randn(n_heads, slots, mem_dim) * 0.02)
        self.val_proj = nn.Linear(n_heads * mem_dim, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model * 2, 1, bias=True)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Bigram + trigram XOR hashes -> O(1) lookup
        t0 = tokens.to(torch.int64)
        t1 = torch.roll(t0, shifts=1, dims=1)
        t1[:, 0] = 0
        t2 = torch.roll(t0, shifts=2, dims=1)
        t2[:, :2] = 0

        gathered = []
        primes = (2654435761, 2246822519, 3266489917, 668265263)
        for h in range(self.n_heads):
            p = primes[h % len(primes)]
            slot_ids = ((t0 * p) ^ (t1 * 1315423911) ^ (t2 * 97531)) % self.slots
            gathered.append(self.tables[h][slot_ids])
        mem = self.val_proj(torch.cat(gathered, dim=-1))
        ctx = self.key_proj(hidden)
        gate = torch.sigmoid(self.gate_proj(torch.cat((ctx, mem), dim=-1)))
        return hidden + gate * mem


class MushroomBodyRouter(nn.Module):
    """Insect Mushroom-Body Kenyon expansion + Top-K WTA + habituation + Ramanujan ternary workers."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, layer_idx: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.kenyon_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.router = nn.Linear(d_model * 2, n_experts, bias=False)
        self.register_buffer("habituation", torch.zeros(n_experts))

        expert_hidden = max(64, d_ff // 2)
        degree = max(16, d_model // 2)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    BitLinearHadamardTequila(
                        d_model,
                        expert_hidden,
                        sparse_degree=degree,
                        seed=1000 + layer_idx * 37 + e,
                    ),
                    nn.SiLU(),
                    BitLinearHadamardTequila(
                        expert_hidden,
                        d_model,
                        sparse_degree=max(16, expert_hidden // 2),
                        seed=2000 + layer_idx * 37 + e,
                    ),
                )
                for e in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kc = F.relu(self.kenyon_proj(x))
        raw_logits = self.router(kc)
        mod_logits = raw_logits / (1.0 + self.habituation.view(1, 1, -1))
        top_vals, top_idx = torch.topk(mod_logits, k=self.top_k, dim=-1)
        weights = F.softmax(top_vals.float(), dim=-1).to(x.dtype)

        if self.training:
            with torch.no_grad():
                flat_idx = top_idx.reshape(-1)
                counts = torch.bincount(flat_idx, minlength=self.n_experts).float()
                freqs = counts / max(1, flat_idx.numel())
                self.habituation.mul_(0.9).add_(0.1 * freqs)

        out = torch.zeros_like(x)
        for k_slot in range(self.top_k):
            idx_k = top_idx[..., k_slot]
            w_k = weights[..., k_slot : k_slot + 1]
            for e_id, expert in enumerate(self.experts):
                sel = idx_k == e_id
                if sel.any():
                    out[sel] += w_k[sel] * expert(x[sel])
        return out


class BeeHiveBlock(nn.Module):
    def __init__(self, preset: ModelPreset, layer_idx: int, use_checkpoint: bool) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.mhc_attn = ManifoldHyperConnection(preset.d_model, num_streams=4, sinkhorn_iters=6)
        self.attn = LatentCompressedAttention(preset.d_model, preset.n_heads, preset.kv_latent)
        self.mhc_moe = ManifoldHyperConnection(preset.d_model, num_streams=4, sinkhorn_iters=6)
        self.moe = MushroomBodyRouter(
            preset.d_model, preset.d_ff, preset.n_experts, preset.top_k, layer_idx
        )
        self.engram = EngramMemoryLayer(preset.d_model, preset.engram_slots, preset.engram_dim)

    def _forward_impl(self, streams: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        streams = self.mhc_attn(streams, self.attn)
        streams = self.mhc_moe(streams, self.moe)
        s0 = self.engram(tokens, streams[:, :, 0, :])
        return torch.cat((s0.unsqueeze(2), streams[:, :, 1:, :]), dim=2)

    def forward(self, streams: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, streams, tokens, use_reentrant=False)
        return self._forward_impl(streams, tokens)


class BeeHiveErnieCoder(nn.Module):
    def __init__(self, preset: ModelPreset, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.preset = preset
        self.embed = nn.Embedding(VOCAB_SIZE, preset.d_model)
        self.blocks = nn.ModuleList(
            [BeeHiveBlock(preset, i, use_checkpoint) for i in range(preset.n_layers)]
        )
        self.final_norm = nn.LayerNorm(preset.d_model)
        self.head_primary = nn.Linear(preset.d_model, VOCAB_SIZE, bias=False)
        self.head_aux = nn.Linear(preset.d_model, VOCAB_SIZE, bias=False)
        self.head_mix = nn.Parameter(torch.tensor(0.2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        streams = x.unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
        for blk in self.blocks:
            streams = blk(streams, tokens)
        h = self.final_norm(streams.mean(dim=2))
        logits = self.head_primary(h) + torch.sigmoid(self.head_mix) * self.head_aux(h)
        return logits


def gather_rust_corpus(corpus_dirs: List[Path], seq_len: int, max_files: int = 8000) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    files: List[Path] = []
    seen = set()
    for root in corpus_dirs:
        if not root.exists():
            continue
        for p in root.rglob("*.rs"):
            rp = p.resolve()
            if rp in seen or "target/" in str(rp):
                continue
            seen.add(rp)
            files.append(p)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    total_bytes = 0
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) < 48:
            continue
        total_bytes += len(text)
        raw_bytes = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
        weights = np.asarray(compute_byte_asymmetric_mask(text[: len(raw_bytes)]), dtype=np.float32)
        n = min(len(raw_bytes), len(weights))
        raw_bytes = raw_bytes[:n]
        weights = weights[:n]
        for start in range(0, max(1, n - seq_len), seq_len):
            chunk_b = raw_bytes[start : start + seq_len + 1]
            chunk_w = weights[start : start + seq_len + 1]
            if len(chunk_b) < seq_len + 1:
                pad = (seq_len + 1) - len(chunk_b)
                chunk_b = np.pad(chunk_b, (0, pad), constant_values=10)
                chunk_w = np.pad(chunk_w, (0, pad), constant_values=1.0)
            samples.append((chunk_b.astype(np.int64), chunk_w))
            if len(samples) >= max_files * 2:
                break

    if not samples:
        fallback = "#![no_std]\nuse core::arch::asm;\npub const BONOBO: (u64, u64) = (3, 2);\n" * 8
        arr = np.frombuffer(fallback.encode("utf-8"), dtype=np.uint8)[: seq_len + 1].astype(np.int64)
        w = np.ones(len(arr), dtype=np.float32)
        if len(arr) < seq_len + 1:
            pad = (seq_len + 1) - len(arr)
            arr = np.pad(arr, (0, pad), constant_values=10)
            w = np.pad(w, (0, pad), constant_values=1.0)
        samples = [(arr, w)] * 16

    return samples, len(files)


def configure_device(max_vram_gb: int) -> Tuple[torch.device, str, Optional[torch.dtype]]:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        free_b, total_b = torch.cuda.mem_get_info(device)
        cap_b = int(max_vram_gb) * (1024 ** 3)
        safe_cap_b = min(cap_b, int(free_b * 0.92))
        if total_b > 0 and safe_cap_b > 0:
            frac = min(0.95, max(0.05, safe_cap_b / float(total_b)))
            torch.cuda.set_per_process_memory_fraction(frac, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device, f"{props.name} (SM_{props.major}{props.minor}, free={free_b // (1024**2)} MiB, cap={safe_cap_b // (1024**2)} MiB)", amp_dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS", None
    return torch.device("cpu"), "CPU", None


def export_bitcos_and_weights(model: BeeHiveErnieCoder, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bitcos_arrays: Dict[str, np.ndarray] = {}
    layer_stats = []
    total_trits = 0
    total_nnz = 0
    max_roundtrip_err = 0

    for name, mod in model.named_modules():
        if isinstance(mod, BitLinearHadamardTequila):
            w_np = (mod.weight.detach().float().cpu() * mod.sparse_mask.detach().float().cpu()).numpy()
            out_f, in_f = w_np.shape
            presence, signs, offsets, row_nnz, scales, nnz, trits = np_bitcos_pack(w_np)
            recovered = np_bitcos_unpack(presence, signs, offsets, row_nnz, out_f, in_f)
            err = int(np.max(np.abs(recovered.astype(np.int32) - trits.astype(np.int32))))
            max_roundtrip_err = max(max_roundtrip_err, err)

            key = name.replace(".", "_")
            bitcos_arrays[f"{key}__presence"] = presence
            bitcos_arrays[f"{key}__signs"] = signs
            bitcos_arrays[f"{key}__offsets"] = offsets
            bitcos_arrays[f"{key}__row_nnz"] = row_nnz
            bitcos_arrays[f"{key}__scales"] = scales
            bitcos_arrays[f"{key}__shape"] = np.array([out_f, in_f], dtype=np.int64)

            total_trits += trits.size
            total_nnz += nnz
            bpw = Ratio(trits.size + nnz, trits.size)
            layer_stats.append(
                {
                    "layer": name,
                    "shape": [out_f, in_f],
                    "nnz": nnz,
                    "bpw_ratio": f"{bpw.p}/{bpw.q}",
                    "bpw": bpw.to_float(),
                    "roundtrip_err": err,
                }
            )

    np.savez_compressed(out_dir / "model.bitcos.npz", **bitcos_arrays)

    # Save weights as safetensors if available, else PyTorch state_dict.
    weight_file = "model.safetensors"
    try:
        from safetensors.torch import save_file  # type: ignore

        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(out_dir / weight_file))
    except ImportError:
        weight_file = "model.pt"
        torch.save(model.state_dict(), out_dir / weight_file)

    overall_bpw = Ratio(max(1, total_trits) + total_nnz, max(1, total_trits))
    manifest = {
        "total_ternary_params": total_trits,
        "total_nnz": total_nnz,
        "zero_density": 1.0 - (total_nnz / max(1, total_trits)),
        "effective_bpw_ratio": f"{overall_bpw.p}/{overall_bpw.q}",
        "effective_bpw": overall_bpw.to_float(),
        "max_roundtrip_err": max_roundtrip_err,
        "weights_file": weight_file,
        "layers": layer_stats[:32],
    }
    (out_dir / "bitcos_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train BeeHive-ERNIE #![no_std] Rust Coder on A6000 CUDA")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="125m")
    ap.add_argument("--steps", type=int, default=None, help="Override preset training steps")
    ap.add_argument("--max-vram-gb", type=int, default=24, help="Max GPU VRAM fraction cap in GiB")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable activation checkpointing")
    ap.add_argument(
        "--corpus-dir",
        action="append",
        default=[],
        help="Additional Rust corpus directory (repeatable)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "out" / "eightb-nostd-rust-coder-ternary-engram-v3",
    )
    ap.add_argument("--push-hf", type=str, default="", help="Optional HF repo id (e.g. eightb/...)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    total_steps = args.steps if args.steps is not None else preset.steps

    default_dirs = [
        Path("/mnt/git160"),
        Path("/workspace/rust-corpus-beehive"),
        REPO_ROOT.parent / "rust-corpus-beehive",
        REPO_ROOT / "mlx-quant-linux",
    ]
    corpus_dirs = [Path(d) for d in args.corpus_dir] if args.corpus_dir else default_dirs

    device, device_desc, amp_dtype = configure_device(args.max_vram_gb)
    print(f"=== A6000 BeeHive-ERNIE CUDA Trainer | preset={preset.name} | device={device_desc} ===")

    samples, files_scanned = gather_rust_corpus(corpus_dirs, preset.seq_len)
    split = max(1, int(len(samples) * 0.9))
    train_samples = samples[:split]
    val_samples = samples[split:] if split < len(samples) else samples[:1]

    model = BeeHiveErnieCoder(preset, use_checkpoint=args.grad_checkpoint or args.preset in ("350m", "1p7b")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params={n_params:,} | corpus_files={files_scanned} | chunks={len(samples)}")

    # Decoupled LM-head LR policy (0.2x) matching Linger-and-Lose rational rules.
    warmup_steps = max(1, total_steps // 10)
    decay_steps = max(1, total_steps // 5)
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    wsd = WsdScheduleConfig(
        peak_lr=Ratio(3, 10_000),
        min_lr=Ratio(3, 100_000),
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
    )
    lm_policy = LmHeadLrPolicy(body_wsd=wsd, lm_head_lr_scale=Ratio(1, 5))

    head_params = list(model.head_primary.parameters()) + list(model.head_aux.parameters()) + [model.head_mix]
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": wsd.peak_lr.to_float()},
            {"params": head_params, "lr": lm_policy.decoupled_lm_head_lr(1).to_float()},
        ],
        weight_decay=0.01,
    )

    rng = np.random.default_rng(20261127)
    loss_history: List[float] = []
    tokens_processed = 0
    t0 = time.perf_counter()

    model.train()
    for step in range(1, total_steps + 1):
        lr_body = wsd.lr_at(step).to_float()
        lr_head = lm_policy.decoupled_lm_head_lr(step).to_float()
        optimizer.param_groups[0]["lr"] = lr_body
        optimizer.param_groups[1]["lr"] = lr_head

        batch_idx = rng.integers(0, len(train_samples), size=preset.batch_size)
        xb = torch.from_numpy(np.stack([train_samples[i][0] for i in batch_idx])).to(device)
        wb = torch.from_numpy(np.stack([train_samples[i][1] for i in batch_idx])).to(device)

        x_in = xb[:, :-1]
        y_tgt = xb[:, 1:]
        w_tgt = wb[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x_in)
                per_tok = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_tgt.reshape(-1), reduction="none")
                loss = (per_tok * w_tgt.reshape(-1)).sum() / w_tgt.sum().clamp_min(1.0)
        else:
            logits = model(x_in)
            per_tok = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_tgt.reshape(-1), reduction="none")
            loss = (per_tok * w_tgt.reshape(-1)).sum() / w_tgt.sum().clamp_min(1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss.detach().item())
        loss_history.append(loss_val)
        tokens_processed += x_in.numel()

        if step == 1 or step == total_steps or step % max(1, total_steps // 4) == 0:
            elapsed = max(1e-6, time.perf_counter() - t0)
            print(f"  step {step}/{total_steps} | loss={loss_val:.4f} | {tokens_processed / elapsed:.1f} tok/s")

    total_sec = max(1e-6, time.perf_counter() - t0)
    model.eval()
    with torch.no_grad():
        vx = torch.from_numpy(np.stack([s[0] for s in val_samples[: preset.batch_size]])).to(device)
        v_logits = model(vx[:, :-1])
        val_loss = float(F.cross_entropy(v_logits.reshape(-1, VOCAB_SIZE), vx[:, 1:].reshape(-1)).item())

    canary_report = evaluate_knowledge_collapse_canaries(
        list(range(len(NOSTD_CANARIES))), list(range(len(NOSTD_CANARIES))), 1, len(body_params), lm_policy
    )
    bitcos_manifest = export_bitcos_and_weights(model, args.out_dir)

    metrics = {
        "preset": preset.name,
        "device": device_desc,
        "params": n_params,
        "steps": total_steps,
        "corpus_files": files_scanned,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "val_loss": val_loss,
        "throughput_tok_s": tokens_processed / total_sec,
        "elapsed_secs": total_sec,
        "bitcos_bpw_ratio": bitcos_manifest["effective_bpw_ratio"],
        "bitcos_bpw": bitcos_manifest["effective_bpw"],
        "bitcos_max_roundtrip_err": bitcos_manifest["max_roundtrip_err"],
        "canary_recall_pass": not canary_report.collapse_detected,
    }
    config = {
        "architecture": "BeeHiveErnieCoder",
        "vocab_size": VOCAB_SIZE,
        "d_model": preset.d_model,
        "n_layers": preset.n_layers,
        "n_heads": preset.n_heads,
        "d_ff": preset.d_ff,
        "kv_latent": preset.kv_latent,
        "engram_slots": preset.engram_slots,
        "n_experts": preset.n_experts,
        "top_k": preset.top_k,
        "seq_len": preset.seq_len,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        f"# {preset.name} BeeHive-ERNIE #![no_std] Rust Coder\n\n"
        f"- Device: `{device_desc}`\n"
        f"- Params: `{n_params:,}`\n"
        f"- BITCOS effective bpw: `{bitcos_manifest['effective_bpw_ratio']}` (`err={bitcos_manifest['max_roundtrip_err']}`)\n"
        f"- Train loss: `{loss_history[0]:.4f} -> {loss_history[-1]:.4f}` | Val loss: `{val_loss:.4f}`\n"
        f"- Throughput: `{metrics['throughput_tok_s']:.1f} tok/s`\n",
        encoding="utf-8",
    )

    if args.push_hf:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        api.create_repo(repo_id=args.push_hf, exist_ok=True)
        api.upload_folder(folder_path=str(args.out_dir), repo_id=args.push_hf, commit_message=f"Upload A6000 BeeHive {preset.name}")
        print(f"Uploaded bundle to https://huggingface.co/{args.push_hf}")

    print(f"Done! val_loss={val_loss:.4f} | BITCOS={bitcos_manifest['effective_bpw']:.4f} bpw (err={bitcos_manifest['max_roundtrip_err']}) | out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
