#!/usr/bin/env python3
"""
E2E Training & ultraSOTAFineTuning:
`8b-TernaryForge v2` -> `eightb/nostd-rust-coder-ternary-engram-v1`
===================================================================
Runs natively on Apple Silicon Metal GPU using MLX-QUANT v2.1.0 ("Zeno-Engram"):
  1. Teacher Warm-Up (Phase 0): dense SubLN + ReLU^2 + Engram causal LM on local #![no_std] Rust.
  2. BitDistill + Tequila + H-BitLinear QAT (Phase 1 & 2):
       - H-BitLinear (`use_hadamard=True`, `act_bits=4`, W1.58A4, arXiv:2504.18415)
       - Tequila trapping-free deadzone dynamic-bias STE (`tequila_lambda=1e-3`, arXiv:2509.23809)
       - DeepSeek Engram O(1) multi-head N-gram conditional memory (`EngramMemoryLayer`, arXiv:2601.07372)
       - Dual-Signal BitDistill loss (3/1 Asymmetric CE + Teacher Logit KL, arXiv:2510.13998)
       - Exact p/q WSD `OneMinusSqrt` cooldown + Decoupled 0.2x LM-Head LR (`LmHeadLrPolicy`, arXiv:2609.32902)
  3. Sub-1.58-Bit BITCOS Export (`arXiv:2609.16338`):
       - Packs all ternary BitLinear layers into presence bitmap + compacted signs (< 1.585 bpw)
       - Evaluates Linger-and-Lose exact canary recall & held-out validation loss
       - Generates complete Hugging Face Model Card + artifacts in `out/eightb-nostd-rust-coder-ternary-engram-v1/`.
"""

from functools import partial
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_flatten, tree_map

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import mlx.quant as mq  # noqa: E402
from mlx.nn.layers.bitlinear import BitLinear  # noqa: E402
from mlx.quant import (  # noqa: E402
    ASYMMETRIC_CRITICAL_WEIGHT,
    EngramMemoryLayer,
    LmHeadLrPolicy,
    Ratio,
    WsdScheduleConfig,
    bitcos_pack_numpy,
    bitcos_unpack_numpy,
    compute_byte_asymmetric_mask,
    evaluate_knowledge_collapse_canaries,
)

OMNILLL_ROOT = Path(
    "/Users/peterlodri/workspace/peterlodri-sec/training/OmniLLL_Liberate_Literate_Love"
)
LOCAL_RUST_DIRS = [
    OMNILLL_ROOT / "crates" / "omnilll-core" / "src",
    OMNILLL_ROOT / "crates" / "omnilll-gpu" / "src",
    OMNILLL_ROOT / "crates" / "omnilll-verify" / "src",
    Path("/Users/peterlodri/workspace/peterlodri-sec/smart-tree/src"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/8xtract/src"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/OneCompression/src"),
]
OUT_DIR = REPO_ROOT / "out" / "eightb-nostd-rust-coder-ternary-engram-v1"

# Model hyperparameters (power-of-2 d_model & d_ff for orthonormal Hadamard WHT)
VOCAB_SIZE = 256
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
D_FF = 1024
MAX_SEQ_LEN = 128
THRESHOLD = 0.76
TEQUILA_LAMBDA = 1e-3
ENGRAM_LAYERS = (0, 3)
ENGRAM_TABLE_SIZE = 4099
ENGRAM_HEADS = 4
ENGRAM_HEAD_DIM = 16


# ---------------------------------------------------------------------------
# 1. Model Definition (Dense Teacher & Ternary+Engram Student)
# ---------------------------------------------------------------------------


def make_proj(in_dims: int, out_dims: int, ternary: bool) -> nn.Module:
    if ternary:
        return BitLinear(
            in_dims,
            out_dims,
            bias=False,
            deployed_forward=True,
            tequila_lambda=TEQUILA_LAMBDA,
            use_hadamard=True,
            act_bits=4,
            threshold=THRESHOLD,
        )
    return nn.Linear(in_dims, out_dims, bias=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ternary: bool) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = make_proj(d_model, d_model, ternary)
        self.k_proj = make_proj(d_model, d_model, ternary)
        self.v_proj = make_proj(d_model, d_model, ternary)
        self.sub_ln = nn.RMSNorm(d_model, eps=1e-6)
        self.o_proj = make_proj(d_model, d_model, ternary)
        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask="causal"
        )
        merged = attn.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)
        return self.o_proj(self.sub_ln(merged))


class ReLU2FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, ternary: bool) -> None:
        super().__init__()
        self.up_proj = make_proj(d_model, d_ff, ternary)
        self.sub_ln = nn.RMSNorm(d_ff, eps=1e-6)
        self.down_proj = make_proj(d_ff, d_model, ternary)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.up_proj(x))
        h = h * h
        return self.down_proj(self.sub_ln(h))


class TernaryEngramBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        ternary: bool,
        with_engram: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.attn = CausalSelfAttention(d_model, n_heads, ternary)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = ReLU2FFN(d_model, d_ff, ternary)
        self.engram = (
            EngramMemoryLayer(
                d_model=d_model,
                vocab_size=VOCAB_SIZE,
                table_size=ENGRAM_TABLE_SIZE,
                num_heads=ENGRAM_HEADS,
                ngram_orders=(2, 3),
                head_dim=ENGRAM_HEAD_DIM,
            )
            if with_engram
            else None
        )

    def __call__(self, x: mx.array, token_ids: mx.array) -> mx.array:
        if self.engram is not None:
            x = x + self.engram(x, token_ids)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TernaryEngramRustCoder(nn.Module):
    def __init__(self, ternary: bool = True) -> None:
        super().__init__()
        self.ternary = ternary
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.embed.weight = mx.random.normal((VOCAB_SIZE, D_MODEL)) * 0.02
        self.blocks = [
            TernaryEngramBlock(
                D_MODEL,
                N_HEADS,
                D_FF,
                ternary=ternary,
                with_engram=(idx in ENGRAM_LAYERS),
            )
            for idx in range(N_LAYERS)
        ]
        self.norm = nn.RMSNorm(D_MODEL, eps=1e-6)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(token_ids)
        for block in self.blocks:
            h = block(h, token_ids)
        return self.lm_head(self.norm(h))


# ---------------------------------------------------------------------------
# 2. Local #![no_std] Rust Corpus + 32 Exact-Recall Canary Probes
# ---------------------------------------------------------------------------


def build_canaries(count: int = 32) -> List[Tuple[str, str, int]]:
    canaries = []
    for i in range(count):
        # Deterministic pseudo-random 32-bit hex constant per canary
        val = ((i * 2654435761) ^ 0x8B150062) & 0xFFFF_FFFF
        prompt = f"// CANARY_{i:02d}: pub const NOSTD_K{i:02d}: u32 = "
        answer = f"0x{val:08X};"
        canaries.append((prompt, answer, val))
    return canaries


def encode_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.uint32)


def decode_bytes(ids: List[int]) -> str:
    return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")


def gather_nostd_rust_corpus(
    canaries: List[Tuple[str, str, int]],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    rust_chunks: List[str] = []
    files_read = 0
    total_bytes = 0

    for d in LOCAL_RUST_DIRS:
        if not d.exists():
            continue
        for rs_path in sorted(d.rglob("*.rs")):
            try:
                text = rs_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(text) < 200:
                continue
            files_read += 1
            total_bytes += len(text)
            # Chunk into clean Rust snippets of at least MAX_SEQ_LEN + 1 bytes
            for start in range(0, min(len(text), 24_000), 256):
                snippet = text[start : start + 512].lstrip("\n")
                if len(snippet.encode("utf-8", errors="ignore")) >= MAX_SEQ_LEN + 1:
                    rust_chunks.append(snippet)

    # Synthetic #![no_std] Rust primitives & SIMD snippets (Anchor Mix)
    curated_nostd = [
        "#![no_std]\nuse core::arch::asm;\n#[inline(always)]\npub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {\n    if a == 0 { return b; }\n    if b == 0 { return a; }\n    let s = (a | b).trailing_zeros();\n    a >>= a.trailing_zeros();\n    loop {\n        b >>= b.trailing_zeros();\n        if a > b { core::mem::swap(&mut a, &mut b); }\n        b -= a;\n        if b == 0 { return a << s; }\n    }\n}\n",
        "#[repr(C)]\npub struct Ratio { pub p: u64, pub q: u64 }\nimpl Ratio {\n    pub const ONE: Self = Self { p: 1, q: 1 };\n    pub const BONOBO: Self = Self { p: 3, q: 2 };\n}\n",
        "#[repr(i8)]\n#[derive(Copy, Clone, PartialEq, Eq)]\npub enum Trit { Neg = -1, Zero = 0, Pos = 1 }\npub const Q16_ONE: i32 = 65536;\n",
    ]
    rust_chunks.extend(curated_nostd * 24)

    # Deterministic 90/10 train/val split for real Rust chunks
    rng = np.random.default_rng(42)
    idx_perm = rng.permutation(len(rust_chunks))
    val_count = max(16, min(64, len(rust_chunks) // 10))
    val_indices = set(idx_perm[:val_count].tolist())

    def to_sample(txt: str) -> Tuple[np.ndarray, np.ndarray]:
        toks = encode_bytes(txt)[: MAX_SEQ_LEN + 1]
        if len(toks) < MAX_SEQ_LEN + 1:
            pad_len = (MAX_SEQ_LEN + 1) - len(toks)
            toks = np.pad(toks, (0, pad_len), constant_values=10)  # '\n'
        w_list = compute_byte_asymmetric_mask(txt)[: MAX_SEQ_LEN + 1]
        if len(w_list) < MAX_SEQ_LEN + 1:
            w_list.extend([1.0] * ((MAX_SEQ_LEN + 1) - len(w_list)))
        return toks, np.asarray(w_list, dtype=np.float32)

    train_rust = [to_sample(rust_chunks[i]) for i in range(len(rust_chunks)) if i not in val_indices]
    val_rust = [to_sample(rust_chunks[i]) for i in range(len(rust_chunks)) if i in val_indices]

    # Pack canary probes into exact-recall blocks (4 canaries per sequence, tiled cleanly)
    canary_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for base in range(0, len(canaries), 2):
        pair = canaries[base : base + 2]
        unit_txt = "#![no_std]\n" + "".join(f"{p}{a}\n" for p, a, _ in pair)
        block_txt = (unit_txt * 3)[: 512]
        toks, w_arr = to_sample(block_txt)
        # Boost critical loss weight on canary hex answers to prioritize factual retention
        w_arr = np.maximum(w_arr, 2.0)
        canary_samples.append((toks, w_arr))

    # Mix 75% Rust + 25% Canary blocks in training pool
    train_pool = list(train_rust)
    repeat_canaries = max(1, (len(train_rust) // 3) // max(1, len(canary_samples)))
    for _ in range(repeat_canaries):
        train_pool.extend(canary_samples)

    stats = {
        "rust_files_scanned": files_read,
        "rust_source_bytes": total_bytes,
        "train_rust_chunks": len(train_rust),
        "val_rust_chunks": len(val_rust),
        "canary_blocks": len(canary_samples),
        "train_pool_size": len(train_pool),
    }
    return train_pool, val_rust, stats


def sample_batch(
    pool: List[Tuple[np.ndarray, np.ndarray]], batch_size: int, rng: np.random.Generator
) -> Tuple[mx.array, mx.array, mx.array]:
    idxs = rng.integers(0, len(pool), size=batch_size)
    toks = np.stack([pool[i][0] for i in idxs], axis=0)
    wts = np.stack([pool[i][1] for i in idxs], axis=0)
    x = mx.array(toks[:, :-1], dtype=mx.uint32)
    y = mx.array(toks[:, 1:], dtype=mx.uint32)
    w = mx.array(wts[:, 1:], dtype=mx.float32)
    return x, y, w


# ---------------------------------------------------------------------------
# 3. Loss Functions, Decoupled Head Grads & Evaluations
# ---------------------------------------------------------------------------


def asymmetric_ce_loss(logits: mx.array, targets: mx.array, token_weights: mx.array) -> mx.array:
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    return mx.sum(ce * token_weights) / mx.maximum(mx.sum(token_weights), 1.0)


def l1_ternary_sparsity_penalty(model: TernaryEngramRustCoder) -> mx.array:
    penalties = []
    for _, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            penalties.append(mx.mean(mx.abs(mod["weight"])))
    if not penalties:
        return mx.array(0.0, dtype=mx.float32)
    return sum(penalties) / len(penalties)


def bitdistill_student_loss(
    student: TernaryEngramRustCoder,
    teacher_logits: mx.array,
    x: mx.array,
    y: mx.array,
    tok_w: mx.array,
    temp: float = 2.0,
    kl_alpha: float = 0.35,
    l1_beta: float = 2e-3,
) -> mx.array:
    s_logits = student(x)
    ce_loss = asymmetric_ce_loss(s_logits, y, tok_w)

    # Top-k logit KL distillation against frozen dense teacher (BitDistill Phase 2)
    p_t = mx.softmax(mx.stop_gradient(teacher_logits) / temp, axis=-1)
    log_p_s = nn.log_softmax(s_logits / temp, axis=-1)
    log_p_t = mx.log(mx.maximum(p_t, 1e-8))
    kl_per_tok = mx.sum(p_t * (log_p_t - log_p_s), axis=-1) * (temp * temp)
    kl_loss = mx.mean(kl_per_tok)

    sparsity = l1_ternary_sparsity_penalty(student)
    return (1.0 - kl_alpha) * ce_loss + kl_alpha * kl_loss + l1_beta * sparsity


def scale_lm_head_grads(grads: dict, scale: float) -> dict:
    """Apply Linger-and-Lose decoupled LM-head LR scaling (`0.2x`) to gradients."""
    out = dict(grads)
    if "lm_head" in out:
        out["lm_head"] = tree_map(lambda g: g * scale, out["lm_head"])
    return out


def eval_validation_loss(model: TernaryEngramRustCoder, val_pool: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    losses = []
    for i in range(0, len(val_pool), 8):
        chunk = val_pool[i : i + 8]
        toks = np.stack([c[0] for c in chunk], axis=0)
        x = mx.array(toks[:, :-1], dtype=mx.uint32)
        y = mx.array(toks[:, 1:], dtype=mx.uint32)
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def greedy_generate(model: TernaryEngramRustCoder, prompt: str, max_new_tokens: int) -> str:
    prefix = "#![no_std]\n" + prompt
    ids = list(encode_bytes(prefix).tolist())
    for _ in range(max_new_tokens):
        ctx = ids[-MAX_SEQ_LEN:]
        x = mx.array([ctx], dtype=mx.uint32)
        logits = model(x)
        next_id = int(mx.argmax(logits[0, -1]).item())
        ids.append(next_id)
    return decode_bytes(ids[len(encode_bytes(prefix)) :])


def eval_canaries(
    model: TernaryEngramRustCoder,
    canaries: List[Tuple[str, str, int]],
    total_params: int,
    policy: LmHeadLrPolicy,
):
    expected_hashes = []
    predicted_hashes = []
    samples_preview = []
    for prompt, answer, val in canaries:
        gen = greedy_generate(model, prompt, max_new_tokens=len(answer))
        expected_hashes.append(val)
        pred_prefix = gen[: len(answer)]
        if pred_prefix == answer:
            predicted_hashes.append(val)
        else:
            predicted_hashes.append(val ^ 0xFFFF_FFFF)
        if len(samples_preview) < 4:
            samples_preview.append({"prompt": prompt, "expected": answer, "generated": pred_prefix})

    report = evaluate_knowledge_collapse_canaries(
        expected_hashes=expected_hashes,
        predicted_hashes=predicted_hashes,
        canary_bits_each=32,
        model_params=total_params,
        policy=policy,
    )
    return report, samples_preview


# ---------------------------------------------------------------------------
# 4. Transfer Teacher -> Ternary Student & BITCOS Export + Model Card
# ---------------------------------------------------------------------------


def transfer_teacher_to_student(
    teacher: TernaryEngramRustCoder, student: TernaryEngramRustCoder
) -> None:
    t_params = dict(tree_flatten(teacher.parameters()))
    student.load_weights(list(t_params.items()), strict=False)
    mx.eval(student.parameters())


def export_bitcos_bundle(student: TernaryEngramRustCoder, out_dir: Path) -> Dict[str, object]:
    npz_payload: Dict[str, np.ndarray] = {}
    layer_stats = []
    total_weights = 0
    total_nnz = 0
    total_bitcos_bytes = 0
    total_2bit_bytes = 0
    max_unpack_err = 0

    for name, mod in student.named_modules():
        if isinstance(mod, BitLinear) and name:
            w_np = np.array(mod["weight"].astype(mx.float32))
            packed = bitcos_pack_numpy(w_np, threshold=THRESHOLD)
            unpacked = bitcos_unpack_numpy(packed)

            # Verify exact roundtrip vs quantized trits
            scales = np.maximum(np.mean(np.abs(w_np), axis=1, keepdims=True), 1e-8)
            ref_trits = np.where(
                np.abs(w_np) < THRESHOLD * scales, 0, np.where(w_np >= 0, 1, -1)
            ).astype(np.int8)
            err = int(np.max(np.abs(unpacked - ref_trits)))
            max_unpack_err = max(max_unpack_err, err)

            safe_name = name.replace(".", "__")
            npz_payload[f"{safe_name}__presence"] = packed.presence
            npz_payload[f"{safe_name}__signs"] = packed.signs
            npz_payload[f"{safe_name}__row_sign_offsets"] = packed.row_sign_offsets
            npz_payload[f"{safe_name}__row_nnz"] = packed.row_nnz
            npz_payload[f"{safe_name}__scales"] = packed.scales

            n_elems = packed.out_features * packed.in_features
            nnz = packed.total_nnz
            b_bytes = int(packed.presence.nbytes + packed.signs.nbytes)
            two_bit_b = packed.out_features * ((packed.in_features + 3) // 4)
            total_weights += n_elems
            total_nnz += nnz
            total_bitcos_bytes += b_bytes
            total_2bit_bytes += two_bit_b

            layer_stats.append(
                {
                    "layer": name,
                    "shape": [packed.out_features, packed.in_features],
                    "nnz": nnz,
                    "zero_density": round(1.0 - (nnz / max(1, n_elems)), 4),
                    "effective_bpw": round(packed.effective_bpw, 4),
                }
            )

    np.savez_compressed(out_dir / "model.bitcos.npz", **npz_payload)
    agg_bpw = (total_weights + total_nnz) / max(1, total_weights)
    agg_zero_density = 1.0 - (total_nnz / max(1, total_weights))
    manifest = {
        "format": "bitcos-v1",
        "arxiv": "2609.16338",
        "threshold": THRESHOLD,
        "total_ternary_weights": total_weights,
        "total_nonzero_weights": total_nnz,
        "aggregate_zero_density": round(agg_zero_density, 5),
        "aggregate_effective_bpw": round(agg_bpw, 5),
        "compression_vs_fp16": round(16.0 / max(agg_bpw, 1e-6), 2),
        "compression_vs_2bit_packed": round(total_2bit_bytes / max(1, total_bitcos_bytes), 3),
        "total_bitcos_bytes": total_bitcos_bytes,
        "roundtrip_max_trit_error": max_unpack_err,
        "layers": layer_stats[:16],
        "layer_count": len(layer_stats),
    }
    (out_dir / "bitcos_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def benchmark_inference_tps(model: TernaryEngramRustCoder) -> float:
    x = mx.ones((1, 64), dtype=mx.uint32)
    for _ in range(3):
        y = model(x)
        mx.eval(y)
    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters):
        y = model(x)
        mx.eval(y)
    dt = time.perf_counter() - t0
    return (iters * 64.0) / max(dt, 1e-6)


def count_params(model: nn.Module) -> int:
    return sum(int(arr.size) for _, arr in tree_flatten(model.parameters()))


def build_model_card(metrics: Dict[str, object]) -> str:
    bitcos = metrics["bitcos"]
    samples_md = "\n".join(
        f"```rust\n// Prompt: {s['prompt']}\n// Expected: {s['expected']}\n// Generated: {s['generated']}\n```"
        for s in metrics["canary_samples"]
    )
    rust_sample = metrics["rust_completion_sample"]
    return f"""---
license: mit
library_name: mlx-quant
pipeline_tag: text-generation
language:
- en
tags:
- bitnet
- b1.58
- ternary
- bitcos
- engram
- rust
- no_std
- mlx
- apple-silicon
- qat
- tequila
- bitdistill
model-index:
- name: nostd-rust-coder-ternary-engram-v1
  results:
  - task:
      type: text-generation
    metrics:
    - name: Held-out Rust Validation Cross-Entropy
      type: cross_entropy
      value: {metrics['student_final_val_loss']}
    - name: Linger-and-Lose Canary Exact Recall
      type: exact_match
      value: {metrics['canary_recall_float']}
    - name: BITCOS Effective Bits-Per-Weight
      type: bits_per_weight
      value: {bitcos['aggregate_effective_bpw']}
---

# `eightb/nostd-rust-coder-ternary-engram-v1`

**1.58-Bit Ternary (`W1.58A4` `H-BitLinear` + `Tequila` + `BITCOS`) + DeepSeek `Engram` Conditional Memory `#![no_std]` Rust Coder** trained end-to-end on Apple Silicon M1 Max 64 GB via **`MLX-QUANT v2.1.0 ("Zeno-Engram")`**.

---

## Architecture & SOTA Techniques

| Component | Paper / Citation | Implementation in `MLX-QUANT v2.1.0` |
| :--- | :--- | :--- |
| **BitNet b1.58 + SubLN + ReLU²** | Ma et al., Apr 2025 (`arXiv:2504.12285`) | `BitLinear` + `SubLN` `RMSNorm` + squared-ReLU FFN (`d_model={D_MODEL}`, `{N_LAYERS}` layers, `{N_HEADS}` heads, `d_ff={D_FF}`). |
| **`H-BitLinear` (`W1.58A4`)** | BitNet v2, Apr 2025 (`arXiv:2504.18415`) | Orthonormal Fast Walsh–Hadamard (`hadamard_wht_mx`) + symmetric INT4 activation quantizer (`activation_quant_int4`). |
| **`Tequila` Trapping-Free QAT** | Sep 2025 (`arXiv:2509.23809`) | Dynamic-bias reactivation (`tequila_lambda={TEQUILA_LAMBDA}`) rescuing deadzone-trapped ternary weights. |
| **`BitDistill` Dual Distillation** | Oct 2025 (`arXiv:2510.13998`) | Phase-0 dense teacher distilled via `0.65 * Asymmetric_CE(3:1) + 0.35 * KL(T=2.0)` into `W1.58A4` student. |
| **DeepSeek `Engram` Memory** | Jan 2026 (`arXiv:2601.07372`) | `EngramMemoryLayer` (`K={ENGRAM_HEADS}` heads, `(2, 3)` N-grams, `table_size={ENGRAM_TABLE_SIZE}`) injected at layers `{list(ENGRAM_LAYERS)}` with context-aware sigmoid gating. |
| **`Linger & Lose` Guard + `p/q` WSD** | Sep 2026 (`arXiv:2609.32902`) | Exact rational `WsdScheduleConfig` (`OneMinusSqrt` cooldown), decoupled `0.2x` (`Ratio(1, 5)`) LM-head LR, and `32`-canary exact-recall probe. |
| **`BITCOS` Sub-1.58-Bit Export** | Intel Labs, Sep 2026 (`arXiv:2609.16338`) | Presence bitmap + compacted sign stream in `model.bitcos.npz` achieving **`{bitcos['aggregate_effective_bpw']}` bpw** (`{bitcos['compression_vs_fp16']}x` vs FP16). |

---

## Measured E2E Training & Compression Results (Apple M1 Max 64 GB Metal GPU)

- **Total Parameters:** `{metrics['total_params']:,}` (`{metrics['ternary_params']:,}` ternary `BitLinear` weights + Engram N-gram tables + FP embeddings/head)
- **Hardware & Runtime:** `{metrics['device']}`, total E2E training time `{metrics['total_train_seconds']} s`
- **Held-out `#![no_std]` Rust Validation Loss:**
  - Teacher (Phase 0 dense): `{metrics['teacher_val_loss']}`
  - Student (QAT Init -> Final after WSD `OneMinusSqrt` cooldown): `{metrics['student_init_val_loss']}` -> **`{metrics['student_final_val_loss']}`**
- **Linger-and-Lose Canary Probe (`arXiv:2609.32902`):** `{metrics['canary_exact_recalls']} / {metrics['canary_total']}` exact recalls (`{metrics['canary_recall_ratio']}`, `collapse_detected={metrics['canary_collapse_detected']}`)
- **`BITCOS` Packed Weight Stats (`model.bitcos.npz`):**
  - Zero density: **`{round(bitcos['aggregate_zero_density'] * 100, 2)}%`**
  - Effective bits/weight: **`{bitcos['aggregate_effective_bpw']}` bpw** (`< 1.585` theoretical equiprobable ternary floor)
  - Compression vs FP16: **`{bitcos['compression_vs_fp16']}x`** (`{bitcos['compression_vs_2bit_packed']}x` vs 2-bit packed)
  - Unpack round-trip max trit error: `{bitcos['roundtrip_max_trit_error']}`
- **Measured M1 Max Prefill/Forward Throughput:** `{metrics['inference_tok_per_sec']} tok/s` (`batch=1, seq=64`, Metal GPU)

### Sample Canary Exact-Recall Outputs

{samples_md}

### Sample `#![no_std]` Rust Continuation

```rust
#![no_std]
pub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {{
{rust_sample}
```

---

## Files Included

- `model.safetensors`: Full `MLX-QUANT v2.1.0` checkpoint (ternary weights + Engram tables + RMSNorm + embeddings/LM head).
- `model.bitcos.npz` & `bitcos_manifest.json`: Sub-1.58-bit `BITCOS` (`arXiv:2609.16338`) packed arrays (`presence`, `signs`, `row_sign_offsets`, `row_nnz`, `scales`).
- `config.json` & `tokenizer.json`: Model geometry & zero-OOV UTF-8 byte tokenizer specification.
- `training_metrics.json`: Complete step-by-step telemetry, validation loss, canary probes, and `BITCOS` breakdown.

Built by **[8b.is](https://g8t.is)** (`@PeetPedro`) using [`MLX-QUANT v2.1.0`](https://github.com/8b-is/MLX-QUANT) and [`OmniLLL_Liberate_Literate_Love`](https://codeberg.org/8b-is/OmniLLL-Liberate-Literate-Love).
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mx.random.seed(20261127)
    rng = np.random.default_rng(20261127)
    t_start = time.perf_counter()

    print("=" * 78)
    print(f"  MLX-QUANT v{mq.__version__} :: E2E Ternary + Engram #![no_std] Rust Coder")
    print(f"  Device: {mx.default_device()} | Output: {OUT_DIR}")
    print("=" * 78)

    canaries = build_canaries(32)
    train_pool, val_pool, corpus_stats = gather_nostd_rust_corpus(canaries)
    print(f"Corpus Stats: {corpus_stats}")

    # Phase 0: Dense Teacher Warm-Up
    teacher = TernaryEngramRustCoder(ternary=False)
    mx.eval(teacher.parameters())
    total_params = count_params(teacher)
    print(f"Model Params: {total_params:,} total")

    teacher_steps = 140
    teacher_opt = opt.AdamW(learning_rate=1.5e-3, weight_decay=0.01)

    def teacher_loss_fn(m, x, y, w):
        return asymmetric_ce_loss(m(x), y, w)

    teacher_vg = nn.value_and_grad(teacher, teacher_loss_fn)
    teacher_state = [teacher.state, teacher_opt.state]

    @partial(mx.compile, inputs=teacher_state, outputs=teacher_state)
    def teacher_step(x, y, w):
        loss, grads = teacher_vg(teacher, x, y, w)
        teacher_opt.update(teacher, grads)
        return loss

    t0_p0 = time.perf_counter()
    for step in range(teacher_steps):
        x, y, w = sample_batch(train_pool, batch_size=12, rng=rng)
        loss = teacher_step(x, y, w)
        mx.eval(teacher_state, loss)
        if step in (0, 69, teacher_steps - 1):
            print(f"  [Phase 0 Teacher] step {step+1:>3}/{teacher_steps} | loss={float(loss.item()):.4f}")

    teacher_val_loss = eval_validation_loss(teacher, val_pool)
    print(f"Teacher Val Loss: {teacher_val_loss:.4f} ({time.perf_counter() - t0_p0:.1f}s)")

    # Phase 1 & 2: BitDistill + Tequila + H-BitLinear QAT + Engram + p/q WSD + 0.2x LM-Head
    student = TernaryEngramRustCoder(ternary=True)
    transfer_teacher_to_student(teacher, student)
    student_init_val = eval_validation_loss(student, val_pool)
    print(f"Student Init Val Loss (post-surgery W1.58A4): {student_init_val:.4f}")

    qat_steps = 360
    wsd = WsdScheduleConfig(
        peak_lr=Ratio(15, 10_000),  # 0.0015
        min_lr=Ratio(15, 100_000),  # 0.00015
        warmup_steps=20,
        stable_steps=240,
        decay_steps=100,
        decay_shape="one_minus_sqrt",
    )
    policy = LmHeadLrPolicy(
        body_wsd=wsd,
        lm_head_lr_scale=Ratio(1, 5),
        collapse_recall_floor=Ratio(4, 5),
    )
    peak_lr_f = wsd.peak_lr.to_float()
    student_opt = opt.AdamW(learning_rate=peak_lr_f, weight_decay=0.01)

    student_vg = nn.value_and_grad(student, bitdistill_student_loss)
    student_state = [student.state, student_opt.state]
    head_scale_f = policy.lm_head_lr_scale.to_float()

    @partial(mx.compile, inputs=teacher.state)
    def teacher_forward(x):
        return teacher(x)

    @partial(mx.compile, inputs=student_state, outputs=student_state)
    def student_step(x, y, w, teacher_logits, lr_mult):
        loss, grads = student_vg(student, teacher_logits, x, y, w)
        grads = tree_map(lambda g: g * lr_mult, grads)
        grads = scale_lm_head_grads(grads, head_scale_f)
        student_opt.update(student, grads)
        return loss

    curve = []
    t0_p1 = time.perf_counter()

    for step in range(qat_steps):
        lr_ratio = wsd.lr_at(step)
        head_lr_ratio = policy.decoupled_lm_head_lr(step)
        lr_mult = mx.array(lr_ratio.to_float() / peak_lr_f, dtype=mx.float32)

        x, y, w = sample_batch(train_pool, batch_size=12, rng=rng)
        t_logits = teacher_forward(x)
        loss = student_step(x, y, w, t_logits, lr_mult)
        mx.eval(student_state, loss)

        loss_val = float(loss.item())
        if step in (0, 60, 140, 259, 310, qat_steps - 1):
            phase = wsd.phase_at(step)
            print(
                f"  [Phase 1/2 QAT {phase:<6}] step {step+1:>3}/{qat_steps} | "
                f"body_lr={lr_ratio} head_lr={head_lr_ratio} | loss={loss_val:.4f}"
            )
            curve.append(
                {
                    "step": step + 1,
                    "phase": phase,
                    "body_lr": str(lr_ratio),
                    "lm_head_lr": str(head_lr_ratio),
                    "loss": round(loss_val, 5),
                }
            )

    student_final_val = eval_validation_loss(student, val_pool)
    print(f"Student Final Val Loss: {student_final_val:.4f} ({time.perf_counter() - t0_p1:.1f}s)")

    # Canary Recall Evaluation (Linger and Lose probe)
    canary_report, canary_samples = eval_canaries(student, canaries, total_params, policy)
    print(
        f"Canary Probe: {canary_report.exact_recalls}/{canary_report.total_canaries} "
        f"({canary_report.recall_ratio}) | collapse={canary_report.collapse_detected} "
        f"| action={canary_report.recommended_action}"
    )

    # Sample Rust generation
    rust_sample = greedy_generate(
        student, "pub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {\n", max_new_tokens=96
    )

    # Export safetensors + BITCOS + config + tokenizer + metrics + Model Card
    flat_weights = dict(tree_flatten(student.parameters()))
    mx.save_safetensors(str(OUT_DIR / "model.safetensors"), flat_weights)

    bitcos_manifest = export_bitcos_bundle(student, OUT_DIR)
    print(
        f"BITCOS Export: zero_density={bitcos_manifest['aggregate_zero_density']:.4f}, "
        f"effective_bpw={bitcos_manifest['aggregate_effective_bpw']}, "
        f"comp_vs_fp16={bitcos_manifest['compression_vs_fp16']}x, "
        f"err={bitcos_manifest['roundtrip_max_trit_error']}"
    )

    tps = benchmark_inference_tps(student)
    ternary_params = sum(
        int(m["weight"].size) for _, m in student.named_modules() if isinstance(m, BitLinear)
    )
    total_time = round(time.perf_counter() - t_start, 2)

    config_json = {
        "architectures": ["TernaryEngramRustCoder"],
        "model_type": "bitnet-engram-rust-coder",
        "mlx_quant_version": mq.__version__,
        "vocab_size": VOCAB_SIZE,
        "hidden_size": D_MODEL,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "intermediate_size": D_FF,
        "max_position_embeddings": MAX_SEQ_LEN,
        "quantization": {
            "scheme": "bitcos-w1.58a4-tequila-engram",
            "weight_bits": 1.585,
            "effective_bpw": bitcos_manifest["aggregate_effective_bpw"],
            "act_bits": 4,
            "hadamard_wht": True,
            "tequila_lambda": TEQUILA_LAMBDA,
            "threshold": THRESHOLD,
        },
        "engram": {
            "layers": list(ENGRAM_LAYERS),
            "table_size": ENGRAM_TABLE_SIZE,
            "num_heads": ENGRAM_HEADS,
            "ngram_orders": [2, 3],
            "head_dim": ENGRAM_HEAD_DIM,
        },
        "wsd_schedule": {
            "peak_lr": str(wsd.peak_lr),
            "min_lr": str(wsd.min_lr),
            "decay_shape": wsd.decay_shape,
            "lm_head_lr_scale": str(policy.lm_head_lr_scale),
            "asymmetric_critical_weight": str(ASYMMETRIC_CRITICAL_WEIGHT),
        },
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    tokenizer_json = {
        "type": "byte-level-utf8-rust-zero-oov",
        "vocab_size": VOCAB_SIZE,
        "canonical_folding_engram": "ascii-uppercase-to-lowercase + whitespace-collapse",
    }
    (OUT_DIR / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2), encoding="utf-8")

    metrics = {
        "mlx_quant_version": mq.__version__,
        "device": str(mx.default_device()),
        "total_train_seconds": total_time,
        "total_params": total_params,
        "ternary_params": ternary_params,
        "corpus_stats": corpus_stats,
        "teacher_val_loss": round(teacher_val_loss, 4),
        "student_init_val_loss": round(student_init_val, 4),
        "student_final_val_loss": round(student_final_val, 4),
        "canary_total": canary_report.total_canaries,
        "canary_exact_recalls": canary_report.exact_recalls,
        "canary_recall_ratio": str(canary_report.recall_ratio),
        "canary_recall_float": round(canary_report.recall_ratio.to_float(), 4),
        "canary_collapse_detected": canary_report.collapse_detected,
        "canary_action": canary_report.recommended_action,
        "inference_tok_per_sec": round(tps, 1),
        "curve": curve,
        "canary_samples": canary_samples,
        "rust_completion_sample": rust_sample,
        "bitcos": bitcos_manifest,
    }
    (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(build_model_card(metrics), encoding="utf-8")

    print(f"\nExported complete bundle to {OUT_DIR} in {total_time:.1f}s")


if __name__ == "__main__":
    main()
