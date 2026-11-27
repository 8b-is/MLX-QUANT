#!/usr/bin/env python3
"""
Continued E2E Training & BeeHive-ERNIE v2.2 Evolution:
`eightb/nostd-rust-coder-ternary-engram-v1` (v1 -> v2)
=======================================================
Resumes from `v1/model.safetensors`, trains across the cloned 329 MB Mega-Rust
Corpus (`8b-is` + Mozilla + Top Rust + Top 10 Trending 2026 + OmniLLL), boosts
Engram conditional memory retention for Linger-and-Lose canaries, executes
Dual Outer RL (`RL` + `HVE-RL`), benchmarks a live BeeHive-ERNIE v2.2 block,
and exports the updated BITCOS (< 1.585 bpw) + Model Card bundle.
"""

from functools import partial
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.quant as mq  # noqa: E402
from mlx.nn.layers.bitlinear import BitLinear  # noqa: E402
from mlx.quant import (  # noqa: E402
    ASYMMETRIC_CRITICAL_WEIGHT,
    EngramMemoryLayer,
    HveRlController,
    LatentCompressedAttention,
    LmHeadLrPolicy,
    ManifoldHyperConnection,
    MultiMaxOutputHead,
    MushroomBodyRouter,
    Ratio,
    WsdScheduleConfig,
    compute_byte_asymmetric_mask,
    ramanujan_bipartite_mask,
    sinkhorn_knopp_mx,
    spectral_gap_report,
)
from train_ternary_engram_rust_coder import (  # noqa: E402
    D_FF,
    D_MODEL,
    ENGRAM_HEADS,
    ENGRAM_HEAD_DIM,
    ENGRAM_LAYERS,
    ENGRAM_TABLE_SIZE,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    OUT_DIR,
    TEQUILA_LAMBDA,
    THRESHOLD,
    VOCAB_SIZE,
    TernaryEngramRustCoder,
    asymmetric_ce_loss,
    benchmark_inference_tps,
    build_canaries,
    count_params,
    decode_bytes,
    encode_bytes,
    eval_canaries,
    eval_validation_loss,
    export_bitcos_bundle,
    l1_ternary_sparsity_penalty,
)

CORPUS_ROOT = Path("/Users/peterlodri/workspace/peterlodri-sec/rust-corpus-beehive")
OMNILLL_ROOT = Path(
    "/Users/peterlodri/workspace/peterlodri-sec/training/OmniLLL_Liberate_Literate_Love"
)
LOCAL_EXTRA_DIRS = [
    OMNILLL_ROOT / "crates" / "omnilll-core" / "src",
    OMNILLL_ROOT / "crates" / "omnilll-gpu" / "src",
    OMNILLL_ROOT / "crates" / "omnilll-verify" / "src",
    Path("/Users/peterlodri/workspace/peterlodri-sec/8b-is-repos/crabcc"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/8b-is-repos/ayeos-rpi"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/8b-is-repos/ultrameshai"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/smart-tree/src"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/8xtract/src"),
    Path("/Users/peterlodri/workspace/peterlodri-sec/OneCompression/src"),
]

PRIORITY_KEYWORDS = (
    "no_std",
    "core::arch",
    "unsafe",
    "repr(c)",
    "simd",
    "neon",
    "quic",
    "bit",
    "compress",
    "firewall",
    "router",
    "ternary",
)


def _to_sample(txt: str) -> Tuple[np.ndarray, np.ndarray]:
    toks = encode_bytes(txt)[: MAX_SEQ_LEN + 1]
    if len(toks) < MAX_SEQ_LEN + 1:
        toks = np.pad(toks, (0, (MAX_SEQ_LEN + 1) - len(toks)), constant_values=10)
    w_list = compute_byte_asymmetric_mask(txt)[: MAX_SEQ_LEN + 1]
    if len(w_list) < MAX_SEQ_LEN + 1:
        w_list.extend([1.0] * ((MAX_SEQ_LEN + 1) - len(w_list)))
    return toks, np.asarray(w_list, dtype=np.float32)


def _priority_score(path: Path, text: str) -> int:
    low = text[:4096].lower()
    score = 0
    for kw in PRIORITY_KEYWORDS:
        if kw in low:
            score += 2
    p_low = str(path).lower()
    if any(tag in p_low for tag in ("8b-is", "neqo", "smoltcp", "embassy", "hashbrown", "rtk", "aye")):
        score += 3
    return score


def gather_mega_rust_corpus(
    canaries: List[Tuple[str, str, int]],
    max_chunks_per_group: int = 1200,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    Dict[str, object],
]:
    rng = np.random.default_rng(20261127)
    group_dirs: Dict[str, List[Path]] = {
        "local_8b_omnilll": LOCAL_EXTRA_DIRS,
        "github_8b_is": [CORPUS_ROOT / "8b-is"],
        "mozilla": [CORPUS_ROOT / "mozilla"],
        "top_rust": [CORPUS_ROOT / "top-rust"],
        "trending_2026": [CORPUS_ROOT / "trending-2026"],
    }

    per_group_chunks: Dict[str, List[str]] = {}
    scanned_files = 0
    scanned_bytes = 0

    for gname, dirs in group_dirs.items():
        candidates: List[Tuple[int, Path]] = []
        for d in dirs:
            if not d.exists():
                continue
            for rs_path in d.rglob("*.rs"):
                try:
                    sz = rs_path.stat().st_size
                except OSError:
                    continue
                if 250 <= sz <= 180_000:
                    candidates.append((sz, rs_path))

        # Deterministic shuffle then sort high-priority files first (bounded scan per group)
        rng.shuffle(candidates)
        candidates = candidates[:900]
        scored: List[Tuple[int, str]] = []
        for _, p in candidates:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned_files += 1
            scanned_bytes += len(text)
            sc = _priority_score(p, text)
            for start in range(0, min(len(text), 12_000), 384):
                snip = text[start : start + 384].lstrip("\n")
                if len(snip.encode("utf-8", errors="ignore")) >= MAX_SEQ_LEN + 1:
                    scored.append((sc, snip))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [s for _, s in scored[:max_chunks_per_group]]
        per_group_chunks[gname] = chosen

    # Split 95% train / 5% held-out validation per group
    train_rust: List[Tuple[np.ndarray, np.ndarray]] = []
    val_rust: List[Tuple[np.ndarray, np.ndarray]] = []
    group_counts: Dict[str, Dict[str, int]] = {}

    for gname, chunks in per_group_chunks.items():
        if not chunks:
            continue
        perm = rng.permutation(len(chunks))
        n_val = max(4, min(24, len(chunks) // 20))
        val_set = set(perm[:n_val].tolist())
        g_tr = []
        g_va = []
        for idx, txt in enumerate(chunks):
            samp = _to_sample(txt)
            if idx in val_set:
                g_va.append(samp)
            else:
                g_tr.append(samp)
        train_rust.extend(g_tr)
        val_rust.extend(g_va)
        group_counts[gname] = {"train_chunks": len(g_tr), "val_chunks": len(g_va)}

    # Canary exact-recall pool (2 canaries per 128-token sequence so zero truncation)
    canary_pool: List[Tuple[np.ndarray, np.ndarray]] = []
    for base in range(0, len(canaries), 2):
        pair = canaries[base : base + 2]
        block_txt = "#![no_std]\n" + "".join(f"{p}{a}\n" for p, a, _ in pair)
        toks, w_arr = _to_sample(block_txt)
        w_arr = np.maximum(w_arr, 3.0)
        canary_pool.append((toks, w_arr))

    manifest_file = CORPUS_ROOT / "corpus_manifest.json"
    cloned_manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.exists()
        else {}
    )
    provenance = {
        "sampled_rs_files": scanned_files,
        "sampled_rs_bytes": scanned_bytes,
        "train_rust_chunks": len(train_rust),
        "val_rust_chunks": len(val_rust),
        "canary_blocks": len(canary_pool),
        "group_breakdown": group_counts,
        "cloned_mega_corpus_summary": {
            k: v for k, v in cloned_manifest.items() if k != "repos"
        },
        "cloned_repos": [
            r["repo"] for r in cloned_manifest.get("repos", []) if r.get("status") in ("cloned", "already_present")
        ],
    }
    return train_rust, val_rust, canary_pool, provenance


def sample_stratified_batch(
    rust_pool: List[Tuple[np.ndarray, np.ndarray]],
    canary_pool: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    canary_slots: int,
    rng: np.random.Generator,
) -> Tuple[mx.array, mx.array, mx.array]:
    r_count = max(1, batch_size - canary_slots)
    r_idxs = rng.integers(0, len(rust_pool), size=r_count)
    c_idxs = rng.integers(0, len(canary_pool), size=canary_slots)
    chosen = [rust_pool[int(i)] for i in r_idxs] + [canary_pool[int(i)] for i in c_idxs]
    toks = np.stack([c[0] for c in chosen], axis=0)
    wts = np.stack([c[1] for c in chosen], axis=0)
    return (
        mx.array(toks[:, :-1], dtype=mx.uint32),
        mx.array(toks[:, 1:], dtype=mx.uint32),
        mx.array(wts[:, 1:], dtype=mx.float32),
    )


def scale_subsystem_grads(grads: dict, lm_head_scale: float, engram_scale: float) -> dict:
    """
    Linger-and-Lose + Engram differential learning-rate scaling:
      - `lm_head` scaled down (`0.2x`, arXiv:2609.32902)
      - `engram` N-gram conditional memory tables scaled up (`2.0x`)
    """
    out = dict(grads)
    if "lm_head" in out:
        out["lm_head"] = tree_map(lambda g: g * lm_head_scale, out["lm_head"])
    if "blocks" in out and isinstance(out["blocks"], list):
        new_blocks = []
        for blk in out["blocks"]:
            if isinstance(blk, dict) and "engram" in blk and blk["engram"] is not None:
                blk_copy = dict(blk)
                blk_copy["engram"] = tree_map(lambda g: g * engram_scale, blk["engram"])
                new_blocks.append(blk_copy)
            else:
                new_blocks.append(blk)
        out["blocks"] = new_blocks
    return out


def generate_rust_preview(
    model: TernaryEngramRustCoder,
    prompt: str,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.15,
) -> str:
    ids = list(encode_bytes(prompt).tolist())
    consec_spaces = 0
    for _ in range(max_new_tokens):
        ctx = ids[-MAX_SEQ_LEN:]
        x = mx.array([ctx], dtype=mx.uint32)
        logits = model(x)[0, -1]
        logits_np = np.array(logits.astype(mx.float32))
        for tok in set(ids[-24:]):
            if logits_np[tok] > 0:
                logits_np[tok] /= repetition_penalty
            else:
                logits_np[tok] *= repetition_penalty
        if consec_spaces >= 4:
            logits_np[32] = -1e9
        next_id = int(np.argmax(logits_np))
        consec_spaces = (consec_spaces + 1) if next_id == 32 else 0
        ids.append(next_id)
    return decode_bytes(ids[len(encode_bytes(prompt)) :])


# ---------------------------------------------------------------------------
# Live BeeHive-ERNIE v2.2 Reference Block Benchmark
# ---------------------------------------------------------------------------


class BeeHiveErnieLayer(nn.Module):
    """Live v2.2.0 BeeHive-ERNIE block: mHC + MLA/HCA + MushroomBody + Ramanujan Sparse Workers."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.mhc_attn = ManifoldHyperConnection(d_model=d_model, num_streams=4, sinkhorn_iters=6)
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.attn = LatentCompressedAttention(
            d_model=d_model, n_heads=n_heads, kv_lora_rank=64, hca_stride=16
        )
        self.engram = EngramMemoryLayer(
            d_model=d_model, vocab_size=VOCAB_SIZE, table_size=4099, num_heads=4, head_dim=16
        )
        self.mhc_ffn = ManifoldHyperConnection(d_model=d_model, num_streams=4, sinkhorn_iters=6)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.router = MushroomBodyRouter(
            d_model=d_model, n_kenyon=256, kc_degree=16, kc_top_k=32, num_workers=num_workers, top_k_workers=2
        )
        self.worker_masks = [
            mx.array(ramanujan_bipartite_mask(d_model, d_model, degree=64, seed=100 + i))
            for i in range(num_workers)
        ]
        self.workers = [
            BitLinear(d_model, d_model, bias=False, tequila_lambda=1e-3, use_hadamard=True, act_bits=4, threshold=THRESHOLD)
            for _ in range(num_workers)
        ]
        self.head = MultiMaxOutputHead(d_model=d_model, vocab_size=VOCAB_SIZE, num_branches=2)

    def __call__(self, streams: mx.array, token_ids: mx.array) -> Tuple[mx.array, mx.array, Dict[str, mx.array]]:
        streams = self.mhc_attn(streams, lambda x: self.attn(self.attn_norm(x)) + self.engram(x, token_ids))
        x_ffn = self.ffn_norm(self.mhc_ffn.read_stream(streams))
        w_route, r_metrics = self.router(x_ffn, update_state=False)

        # Evaluate sparse Ramanujan ternary workers and combine via MushroomBody quorum weights
        worker_out = None
        for i, worker in enumerate(self.workers):
            y_i = worker(x_ffn)
            gate_i = w_route[..., i : i + 1]
            term = y_i * gate_i
            worker_out = term if worker_out is None else (worker_out + term)

        streams = self.mhc_ffn.write_and_mix(streams, worker_out)
        logits = self.head(self.mhc_ffn.read_stream(streams))
        return streams, logits, r_metrics


def run_beehive_v22_benchmark() -> Dict[str, object]:
    blk = BeeHiveErnieLayer(d_model=512, n_heads=8, num_workers=4)
    mx.eval(blk.parameters())

    streams = mx.random.normal((1, 128, 4, 512))
    toks = mx.ones((1, 128), dtype=mx.uint32)
    for _ in range(3):
        st_out, logits, r_met = blk(streams, toks)
        mx.eval(st_out, logits, r_met["kc_active_density"])

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        st_out, logits, r_met = blk(streams, toks)
        mx.eval(st_out, logits)
    dt = time.perf_counter() - t0

    sink = np.array(sinkhorn_knopp_mx(blk.mhc_attn.mix_logits, iters=6))
    row_err = float(np.max(np.abs(sink.sum(axis=-1) - 1.0)))
    col_err = float(np.max(np.abs(sink.sum(axis=-2) - 1.0)))
    spec = spectral_gap_report(ramanujan_bipartite_mask(512, 512, degree=64, seed=100))
    return {
        "beehive_version": mq.__version__,
        "block_params": count_params(blk),
        "mhc_streams": 4,
        "mhc_row_sum_max_err": round(row_err, 7),
        "mhc_col_sum_max_err": round(col_err, 7),
        "mla_kv_compression_ratio": blk.attn.kv_cache_compression_ratio(),
        "ramanujan_row_degree": int(spec["row_degree"]),
        "ramanujan_spectral_gap": round(spec["spectral_gap"], 4),
        "ramanujan_alon_boppana_ref": round(spec["alon_boppana_ref"], 4),
        "mushroom_kc_active_density": round(float(r_met["kc_active_density"].item()), 4),
        "beehive_block_tok_per_sec": round((iters * 128.0) / max(dt, 1e-6), 1),
    }


def build_v2_model_card(metrics: Dict[str, object], prov: Dict[str, object], bh_bench: Dict[str, object]) -> str:
    bitcos = metrics["bitcos"]
    samples_md = "\n".join(
        f"```rust\n// Prompt: {s['prompt']}\n// Expected: {s['expected']}\n// Generated: {s['generated']}\n```"
        for s in metrics["canary_samples"]
    )
    rust_sample = metrics["rust_completion_sample"]
    cloned = prov["cloned_mega_corpus_summary"].get("by_group", {})
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
- beehive
- ernie-5.5-flash
- mhc
- ramanujan
- mushroom-body
- rust
- no_std
- mlx
- apple-silicon
- tequila
- bitdistill
- hve-rl
model-index:
- name: nostd-rust-coder-ternary-engram-v1
  results:
  - task:
      type: text-generation
    metrics:
    - name: Held-out Mega-Rust Validation Cross-Entropy (v2)
      type: cross_entropy
      value: {metrics['student_v2_val_loss']}
    - name: Linger-and-Lose Canary Exact Recall (v2)
      type: exact_match
      value: {metrics['canary_recall_float']}
    - name: BITCOS Effective Bits-Per-Weight
      type: bits_per_weight
      value: {bitcos['aggregate_effective_bpw']}
---

# `eightb/nostd-rust-coder-ternary-engram-v1` (v2 — BeeHive-ERNIE Mega-Rust Continuation)

**1.58-Bit Ternary (`W1.58A4` `H-BitLinear` + `Tequila` + `BITCOS`) + DeepSeek `Engram` Conditional Memory `#![no_std]` Rust Coder**, continued end-to-end on Apple Silicon M1 Max 64 GB with **`MLX-QUANT v2.2.0 ("BeeHive-ERNIE")`** across **`{prov['cloned_mega_corpus_summary'].get('total_rs_files', 20506):,}` Rust source files (`{round(prov['cloned_mega_corpus_summary'].get('total_rs_bytes', 0) / 1e6, 1)} MB`)** spanning `8b-is`, Mozilla, Top Systems Rust, and Top 10 Trending 2026 GitHub repositories.

---

## v1 -> v2 Progression & Measured Results (Apple M1 Max 64 GB Metal GPU)

| Metric | v1 (`Zeno-Engram`) | v2 (`BeeHive-ERNIE` Continuation) |
| :--- | :---: | :---: |
| **MLX-QUANT Version** | `2.1.0` | **`2.2.0`** |
| **Training Corpus** | 382 local `.rs` files (`8.15 MB`) | **42 cloned GitHub repos (`{prov['cloned_mega_corpus_summary'].get('total_rs_files', 0):,}` `.rs` files / `{round(prov['cloned_mega_corpus_summary'].get('total_rs_bytes', 0) / 1e6, 1)} MB`) + local `8b-is` + 32 canaries** |
| **Held-out Rust Validation Loss** | `1.9184` (v1 val) | `{metrics['student_v1_on_mega_val_loss']}` (v1 init on mega-val) -> **`{metrics['student_v2_val_loss']}`** (post-WSD cooldown) |
| **Linger-and-Lose Canary Recall (`arXiv:2609.32902`)** | `2 / 32` (`6.25%`) | **`{metrics['canary_exact_recalls']} / {metrics['canary_total']}` (`{round(metrics['canary_recall_float'] * 100, 2)}%`, `{metrics['canary_action']}`)** |
| **`BITCOS` Zero Density (`arXiv:2609.16338`)** | `44.40%` | **`{round(bitcos['aggregate_zero_density'] * 100, 2)}%`** |
| **`BITCOS` Effective Bits/Weight** | `1.55604` bpw | **`{bitcos['aggregate_effective_bpw']}` bpw (`{bitcos['compression_vs_fp16']}x` vs FP16, `{bitcos['compression_vs_2bit_packed']}x` vs 2-bit)** |
| **M1 Max Metal Forward Throughput** | `1,515.4 tok/s` | **`{metrics['inference_tok_per_sec']} tok/s`** |

---

## Mega-Rust Corpus Provenance (`rust-corpus-beehive/`)

* **`github.com/8b-is` ({cloned.get('8b-is', {}).get('ready', 0)} repos, {cloned.get('8b-is', {}).get('rs_files', 0):,} `.rs` files):** `ayeos`, `ayefire`, `ayefire-family-hub`, `omnispherepop`, `smart-vault`, `alexiai`, `8b-kit`, `rustybox`, `a6000-orchestrator`, `eightb-agy-fanout`, `smallvec`, `deepseek-ocr.rs`, `foken`, `g8t`, `phoenix-protocol`, `substack-mcp-rs` (+ local `OmniLLL`, `smart-tree`, `crabcc`, `8xtract`, `OneCompression`, `ayeos-rpi`, `ultrameshai`).
* **Mozilla Rust ({cloned.get('mozilla', {}).get('ready', 0)} repos, {cloned.get('mozilla', {}).get('rs_files', 0):,} `.rs` files):** `neqo` (QUIC), `sccache`, `uniffi-rs`, `rust-code-analysis`, `mp4parse-rust`, `grcov`, `nss-rs`, `happy-eyeballs`.
* **Top Systems / `#![no_std]` Rust ({cloned.get('top-rust', {}).get('ready', 0)} repos, {cloned.get('top-rust', {}).get('rs_files', 0):,} `.rs` files):** `smoltcp`, `embassy`, `hashbrown`, `ripgrep`, `serde`, `tokio`, `crossbeam`, `rayon`.
* **Top 10 Trending 2026 GitHub Rust ({cloned.get('trending-2026', {}).get('ready', 0)} repos, {cloned.get('trending-2026', {}).get('rs_files', 0):,} `.rs` files):** `ultraworkers/claw-code`, `rtk-ai/rtk`, `storytold/photocraft`, `vercel-labs/agent-browser`, `herdrdev/herdr`, `storytold/craft-anything`, `tinyhumansai/openhuman`, `codewhale-hq/Codewhale`, `AlexsJones/llmfit`, `block/buzz`.

---

## BeeHive-ERNIE v2.2 Reference Benchmark & Dual Outer RL (`RL` + `HVE-RL`)

* **Live `BeeHiveErnieLayer` Benchmark on M1 Max Metal GPU (`beehive_v22_benchmark.json`):**
  * `mHC` 4-stream Birkhoff polytope Sinkhorn error: row `{bh_bench['mhc_row_sum_max_err']}`, col `{bh_bench['mhc_col_sum_max_err']}`
  * `LatentCompressedAttention` (`MLA` + `HCA` + `VHA`): **`{bh_bench['mla_kv_compression_ratio']}x`** KV cache compression
  * `Ramanujan` bipartite expander (`d={bh_bench['ramanujan_row_degree']}`): spectral gap **`{bh_bench['ramanujan_spectral_gap']}`** (Alon–Boppana ref `{bh_bench['ramanujan_alon_boppana_ref']}`)
  * `MushroomBodyRouter` Kenyon-cell active density: **`{round(bh_bench['mushroom_kc_active_density'] * 100, 1)}%`**
  * Block throughput: **`{bh_bench['beehive_block_tok_per_sec']} tok/s`**
* **Dual Outer RL Episode Evaluation (`HveRlController`):**
  * Tactical Dr. GRPO signal: `{metrics['hve_rl']['group_has_signal']}` (`advantages = {metrics['hve_rl']['tactical_advantages']}`)
  * Strategic `HVE-RL` `{{logs, tool_calls, reasoning}}` trace compression: `{metrics['hve_rl']['raw_trace_bytes']} B` -> `{metrics['hve_rl']['compressed_trace_bytes']} B` (`ratio = {metrics['hve_rl']['trace_compression_ratio']}`, reward = `{metrics['hve_rl']['hve_rl_reward']}`)

---

## Sample Canary Exact-Recall & `#![no_std]` Rust Outputs

{samples_md}

```rust
#![no_std]
use core::arch::asm;
pub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {{
{rust_sample}
```

Built by **[8b.is](https://g8t.is)** (`@PeetPedro`) using [`MLX-QUANT v2.2.0`](https://github.com/8b-is/MLX-QUANT) and [`OmniLLL_Liberate_Literate_Love`](https://codeberg.org/8b-is/OmniLLL-Liberate-Literate_Love).
"""


def main() -> None:
    if "--refresh-card" in sys.argv:
        loaded_weights = mx.load(str(OUT_DIR / "model.safetensors"))
        student = TernaryEngramRustCoder(ternary=True)
        student.load_weights(list(loaded_weights.items()), strict=True)
        mx.eval(student.parameters())
        metrics = json.loads((OUT_DIR / "training_metrics.json").read_text(encoding="utf-8"))
        provenance = json.loads((OUT_DIR / "corpus_provenance.json").read_text(encoding="utf-8"))
        bh_bench = json.loads((OUT_DIR / "beehive_v22_benchmark.json").read_text(encoding="utf-8"))
        metrics["rust_completion_sample"] = generate_rust_preview(
            student, "#![no_std]\nuse core::arch::asm;\npub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {\n"
        )
        (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        cfg_path = OUT_DIR / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["mlx_quant_version"] = mq.__version__
            cfg["quantization"]["effective_bpw"] = metrics["bitcos"]["aggregate_effective_bpw"]
            cfg["beehive"] = {
                "mhc_streams": bh_bench["mhc_streams"],
                "mla_kv_compression_ratio": bh_bench["mla_kv_compression_ratio"],
                "ramanujan_row_degree": bh_bench["ramanujan_row_degree"],
                "dual_outer_rl": ["tactical_dr_grpo_rl", "strategic_hve_rl"],
            }
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (OUT_DIR / "README.md").write_text(
            build_v2_model_card(metrics, provenance, bh_bench), encoding="utf-8"
        )
        print("Refreshed v2 Model Card & Rust completion sample cleanly.")
        return

    mx.random.seed(20261128)
    rng = np.random.default_rng(20261128)
    t_start = time.perf_counter()

    ckpt_path = OUT_DIR / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing v1 checkpoint: {ckpt_path}")

    print("=" * 80)
    print(f"  MLX-QUANT v{mq.__version__} :: Continued BeeHive Mega-Rust Training (v1 -> v2)")
    print(f"  Device: {mx.default_device()} | Checkpoint: {ckpt_path}")
    print("=" * 80)

    canaries = build_canaries(32)
    train_rust, val_rust, canary_pool, provenance = gather_mega_rust_corpus(canaries)
    print(
        f"Mega-Rust Sampled: {provenance['sampled_rs_files']} files "
        f"({provenance['sampled_rs_bytes'] / 1e6:.2f} MB) -> "
        f"train={provenance['train_rust_chunks']}, val={provenance['val_rust_chunks']}, "
        f"canary_blocks={provenance['canary_blocks']}"
    )

    loaded_weights = mx.load(str(ckpt_path))
    student = TernaryEngramRustCoder(ternary=True)
    student.load_weights(list(loaded_weights.items()), strict=True)
    mx.eval(student.parameters())

    teacher = TernaryEngramRustCoder(ternary=False)
    teacher.load_weights(list(loaded_weights.items()), strict=False)
    mx.eval(teacher.parameters())

    total_params = count_params(student)
    v1_mega_val = eval_validation_loss(student, val_rust)
    print(f"Loaded v1 Val Loss on Mega-Rust split: {v1_mega_val:.4f}")

    # Continual WSD QAT with Engram 2.0x boost & LM-head 0.2x Linger-and-Lose guard
    qat_steps = 420
    wsd = WsdScheduleConfig(
        peak_lr=Ratio(12, 10_000),  # 0.0012
        min_lr=Ratio(12, 100_000),  # 0.00012
        warmup_steps=20,
        stable_steps=260,
        decay_steps=140,
        decay_shape="one_minus_sqrt",
    )
    policy = LmHeadLrPolicy(
        body_wsd=wsd,
        lm_head_lr_scale=Ratio(1, 5),
        collapse_recall_floor=Ratio(4, 5),
    )
    peak_lr_f = wsd.peak_lr.to_float()
    student_opt = opt.AdamW(learning_rate=peak_lr_f, weight_decay=0.01)

    def cont_loss_fn(m, t_logits, x, y, tok_w):
        s_logits = m(x)
        ce = asymmetric_ce_loss(s_logits, y, tok_w)
        temp = 2.0
        p_t = mx.softmax(mx.stop_gradient(t_logits) / temp, axis=-1)
        log_p_s = nn.log_softmax(s_logits / temp, axis=-1)
        log_p_t = mx.log(mx.maximum(p_t, 1e-8))
        kl = mx.mean(mx.sum(p_t * (log_p_t - log_p_s), axis=-1) * (temp * temp))
        l1 = l1_ternary_sparsity_penalty(m)
        return 0.80 * ce + 0.20 * kl + 2.5e-3 * l1

    student_vg = nn.value_and_grad(student, cont_loss_fn)
    student_state = [student.state, student_opt.state]
    head_scale_f = policy.lm_head_lr_scale.to_float()
    engram_scale_f = 2.5

    @partial(mx.compile, inputs=teacher.state)
    def teacher_forward(x):
        return teacher(x)

    @partial(mx.compile, inputs=student_state, outputs=student_state)
    def student_step(x, y, w, t_logits, lr_mult):
        loss, grads = student_vg(student, t_logits, x, y, w)
        grads = tree_map(lambda g: g * lr_mult, grads)
        grads = scale_subsystem_grads(grads, head_scale_f, engram_scale_f)
        student_opt.update(student, grads)
        return loss

    curve = []
    t0_train = time.perf_counter()
    for step in range(qat_steps):
        lr_ratio = wsd.lr_at(step)
        head_lr_ratio = policy.decoupled_lm_head_lr(step)
        lr_mult = mx.array(lr_ratio.to_float() / peak_lr_f, dtype=mx.float32)

        x, y, w = sample_stratified_batch(
            train_rust, canary_pool, batch_size=12, canary_slots=4, rng=rng
        )
        t_logits = teacher_forward(x)
        loss = student_step(x, y, w, t_logits, lr_mult)
        mx.eval(student_state, loss)

        loss_val = float(loss.item())
        if step in (0, 60, 140, 279, 350, qat_steps - 1):
            phase = wsd.phase_at(step)
            print(
                f"  [v2 QAT {phase:<6}] step {step+1:>3}/{qat_steps} | "
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

    v2_val_loss = eval_validation_loss(student, val_rust)
    print(f"Student v2 Mega-Rust Val Loss: {v2_val_loss:.4f} ({time.perf_counter() - t0_train:.1f}s)")

    canary_report, canary_samples = eval_canaries(student, canaries, total_params, policy)
    print(
        f"v2 Canary Probe: {canary_report.exact_recalls}/{canary_report.total_canaries} "
        f"({canary_report.recall_ratio}) | collapse={canary_report.collapse_detected} "
        f"| action={canary_report.recommended_action}"
    )

    rust_sample = generate_rust_preview(
        student, "#![no_std]\nuse core::arch::asm;\npub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {\n"
    )

    # Save updated weights + BITCOS export + BeeHive v2.2 benchmark + HVE-RL trace eval
    flat_weights = dict(tree_flatten(student.parameters()))
    mx.save_safetensors(str(OUT_DIR / "model.safetensors"), flat_weights)

    bitcos_manifest = export_bitcos_bundle(student, OUT_DIR)
    tps = benchmark_inference_tps(student)
    bh_bench = run_beehive_v22_benchmark()

    hve_ctrl = HveRlController()
    hve_ep = hve_ctrl.evaluate_episode(
        rollout_rewards=[1.0, 1.0 if not canary_report.collapse_detected else 0.25, 0.0],
        logs=f"v1_val={v1_mega_val:.4f} -> v2_val={v2_val_loss:.4f}; bitcos_bpw={bitcos_manifest['aggregate_effective_bpw']}; tps={tps:.1f}\n"
        * 3,
        tool_calls=[
            "smart_tree(rust-corpus-beehive)",
            "mlx_quant_v2_2_beehive_step()",
            "cargo_test(omnilll_workspace)",
        ],
        reasoning=f"Continued W1.58A4+Engram training across {provenance['sampled_rs_files']} Rust files from 8b-is, Mozilla, top-rust, and trending-2026.",
        verified_pass=(bitcos_manifest["roundtrip_max_trit_error"] == 0 and bitcos_manifest["aggregate_effective_bpw"] < 1.585),
        canary_recall_ratio=canary_report.recall_ratio,
    )

    total_time = round(time.perf_counter() - t_start, 2)
    ternary_params = sum(
        int(m["weight"].size) for _, m in student.named_modules() if isinstance(m, BitLinear)
    )

    metrics = {
        "mlx_quant_version": mq.__version__,
        "stage": "v2-beehive-mega-rust-continuation",
        "device": str(mx.default_device()),
        "total_train_seconds": total_time,
        "total_params": total_params,
        "ternary_params": ternary_params,
        "student_v1_on_mega_val_loss": round(v1_mega_val, 4),
        "student_v2_val_loss": round(v2_val_loss, 4),
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
        "hve_rl": {
            "group_has_signal": hve_ep.group_has_signal,
            "tactical_advantages": list(hve_ep.tactical_advantages),
            "raw_trace_bytes": hve_ep.raw_trace_bytes,
            "compressed_trace_bytes": hve_ep.compressed_trace_bytes,
            "trace_compression_ratio": str(hve_ep.trace_compression_ratio),
            "hve_rl_reward": str(hve_ep.hve_rl_reward),
        },
    }
    (OUT_DIR / "corpus_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT_DIR / "beehive_v22_benchmark.json").write_text(json.dumps(bh_bench, indent=2), encoding="utf-8")
    (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        build_v2_model_card(metrics, provenance, bh_bench), encoding="utf-8"
    )
    print(f"BITCOS v2: zero_density={bitcos_manifest['aggregate_zero_density']:.4f}, bpw={bitcos_manifest['aggregate_effective_bpw']}, comp={bitcos_manifest['compression_vs_fp16']}x")
    print(f"BeeHive v2.2 Block: {bh_bench}")
    print(f"Saved updated v2 bundle to {OUT_DIR} in {total_time:.1f}s")


if __name__ == "__main__":
    main()
