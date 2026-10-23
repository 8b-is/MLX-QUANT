#!/usr/bin/env python3
"""
export_quantal_assets.py — emit the non-ternary model pieces (token embedding
matrix + RMSNorm gain vectors) for the quantal ayeOS export so the ternary
BitNet model becomes runnable end-to-end in Rust.

Background
----------
export_quantal_checkpoint.py emits the per-layer ternary matrix files
(mNNN.json) that carry the quantized BitLinear weights (packed codes +
per-group scales). Those alone cannot generate text — the runtime also needs:

  (a) the token embedding matrix ``model.embed_tokens.weight``
      [vocab, hidden]. This MUST come from the trained checkpoint: quantal
      is a continued-trained model, so the stock Qwen embeddings are a
      different model's tensors.
  (b) the (2*n_layers + 1) RMSNorm gain vectors (n_layers x
      {input, post_attention} + final norm).

This script reads the checkpoint safetensors directly (``mlx.load``) and
writes raw binary assets next to the mNNN.json files:

  embeddings.f16 — BF16 -> FP16, raw little-endian bytes, shape [vocab, hidden]
  norms.f32      — BF16 -> FP32, raw little-endian bytes, shape [2*L+1, hidden]
  qk_norms.f32   — Qwen3-only: per-head q_norm/k_norm gains, shape [4*L, head_dim]
                   (2 per layer x 2 of them), emitted only when the checkpoint
                   carries ``model.layers.0.self_attn.q_norm.weight``.
  gains.f32      — per-projection BitLinear RMSNorm gains (the faithful
                   BitNet b1.58 forward). Only emitted when the checkpoint
                   carries ``model.layers.0.self_attn.q_proj.norm.weight``.
                   Row layout per layer i (all f32, concatenated):
                     q, k, v, o, up, gate: [hidden] each
                     down:                 [intermediate]
                   per layer = 6*hidden + intermediate; total
                   = L * (6*hidden + intermediate). Qwen2.5-0.5B: 24 * 10240
                   = 245,760 floats.
  biases.f32     — q/k/v projection biases (the only biased projections in
                   the quantal BitLinear swap). Per layer i: q [hidden],
                   k [kv_heads*head_dim], v [kv_heads*head_dim]. Qwen2.5-0.5B:
                   24 * 1152 = 27,648 floats. Emitted with gains.f32.

Dimensions are DERIVED from the checkpoint weights (no hardcoded architecture),
so the same script serves Qwen2.5-0.5B (896 / 24 layers / 49 norms) and
Qwen3-1.7B (2048 / 28 layers / 57 norms + qk_norms).

The checkpoint stores these tensors in BF16; the conversion to FP16/FP32 is a
numeric dtype cast via mlx (``astype``), NOT a raw bit copy — interpreting raw
BF16 bits as FP16 would give garbage values.

Both asset files are gitignored (embeddings.f16 exceeds GitHub's 100 MB
per-file limit); only their sha256 metadata is committed in index.json.

norms.f32 row ordering (exact; document this in the Rust loader)
----------------------------------------------------------------
    row 2*i   : model.layers[i].input_layernorm.weight         (i = 0..23)
    row 2*i+1 : model.layers[i].post_attention_layernorm.weight (i = 0..23)
    row 48    : model.norm.weight  (final RMSNorm)

i.e. layer 0 input, layer 0 post_attention, layer 1 input, layer 1
post_attention, ..., layer 23 input, layer 23 post_attention, final norm.
49 rows total, 896 fp32 floats per row (hidden_size).

Usage:
  python scripts/export_quantal_assets.py \
    --checkpoint demos/quantal/quantal_model.safetensors \
    --out-dir demos/quantal
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np

# Legacy hardcoded Qwen2.5-0.5B dimensions, kept only as sanity-check defaults.
# The real values are DERIVED from the checkpoint weights in main() so the same
# script serves Qwen2.5-0.5B (896/24) and Qwen3-1.7B (2048/28).
_VOCAB_SIZE = 151936
_HIDDEN_SIZE = 896
_N_LAYERS = 24

EMBED_KEY = "model.embed_tokens.weight"
FINAL_NORM_KEY = "model.norm.weight"
LAYER_NORM_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
QK_NORM_SUFFIXES = ("q_norm.weight", "k_norm.weight")

# Faithful-BitLinear per-projection extras (deployed-forward checkpoints lack
# these; legacy checkpoints carry them).
PROJ_ORDER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
)
PROJ_INPUT_DIM_KEY = {
    "self_attn.q_proj": "hidden",
    "self_attn.k_proj": "hidden",
    "self_attn.v_proj": "hidden",
    "self_attn.o_proj": "hidden",
    "mlp.up_proj": "hidden",
    "mlp.gate_proj": "hidden",
    "mlp.down_proj": "intermediate",
}
BIAS_PROJS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")

EMBED_FNAME = "embeddings.f16"
NORMS_FNAME = "norms.f32"
QK_NORMS_FNAME = "qk_norms.f32"
GAINS_FNAME = "gains.f32"
BIASES_FNAME = "biases.f32"


def derive_dims(w: dict) -> tuple[int, int, int, int]:
    """Derive (vocab, hidden, n_layers, intermediate) from checkpoint keys."""
    emb = w[EMBED_KEY]
    vocab, hidden = int(emb.shape[0]), int(emb.shape[1])
    layer_keys = [
        k
        for k in w
        if k.startswith("model.layers.") and k.endswith("input_layernorm.weight")
    ]
    n_layers = max((int(k.split(".")[2]) for k in layer_keys), default=-1) + 1
    if n_layers < 1:
        sys.exit("error: no model.layers.*.input_layernorm.weight keys in checkpoint")
    up_key = "model.layers.0.mlp.up_proj.weight"
    intermediate = int(w[up_key].shape[0]) if up_key in w else 0
    return vocab, hidden, n_layers, intermediate


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bf16_tensor(path: Path, header: dict, key: str) -> np.ndarray:
    """Read a single BF16 tensor from a safetensors file, reinterpreted as
    F32 via bit-shift.

    ``mx.load`` is NOT safe here: the local mlx fork mis-reads BF16 payloads
    (a BF16 tensor whose real value is ~0.01 loads as ~15424), which would
    silently corrupt every biased projection. The bit-shift (16-bit value into
    the top half of an f32 pattern, then reinterpret) is exact.

    ``data_offsets`` are relative to the data-section start (after the 8-byte
    header-length prefix + header JSON), NOT the file start.
    """
    info = header[key]
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        data_start = 8 + hdr_len
        f.seek(data_start + start)
        raw = f.read(end - start)
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (bits << 16).view(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="trained quantal safetensors")
    p.add_argument(
        "--out-dir",
        required=True,
        help="pocoo demo dir (m*.json + index.json live here)",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.json"
    if not idx_path.exists():
        sys.exit(
            f"error: {idx_path} not found — run export_quantal_checkpoint.py first"
        )

    print(f"  mlx: {getattr(mx, '__version__', 'unknown')}")
    print(f"  metal: {mx.metal.is_available()}")

    print("1. loading checkpoint...")
    w = cast(dict, mx.load(args.checkpoint))
    if EMBED_KEY not in w:
        sys.exit(f"error: '{EMBED_KEY}' not found in checkpoint")
    if FINAL_NORM_KEY not in w:
        sys.exit(f"error: '{FINAL_NORM_KEY}' not found in checkpoint")

    vocab_size, hidden_size, n_layers, intermediate_size = derive_dims(w)
    expected_norms = 2 * n_layers + 1
    print(
        f"   derived: vocab={vocab_size} hidden={hidden_size} layers={n_layers} "
        f"intermediate={intermediate_size} norms={expected_norms}"
    )

    # --- (a) token embeddings -------------------------------------------------
    print("2. exporting embeddings.f16 ...")
    emb = w[EMBED_KEY]
    assert tuple(emb.shape) == (
        vocab_size,
        hidden_size,
    ), f"unexpected embed shape {emb.shape}"
    print(f"   {EMBED_KEY}: shape={emb.shape} dtype={emb.dtype}")
    emb_f16 = emb.astype(mx.float16)
    emb_bytes = np.asarray(emb_f16, dtype="<f2").tobytes()
    emb_path = out / EMBED_FNAME
    emb_path.write_bytes(emb_bytes)
    emb_size = emb_path.stat().st_size
    print(
        f"   wrote {emb_path} ({emb_size:,} bytes, shape [{vocab_size}, {hidden_size}])"
    )

    # --- (b) RMSNorm vectors --------------------------------------------------
    print("3. exporting norms.f32 ...")
    rows = []
    for i in range(n_layers):
        for suffix in LAYER_NORM_KEYS:
            key = f"model.layers.{i}.{suffix}"
            if key not in w:
                sys.exit(f"error: '{key}' not found in checkpoint")
            rows.append(w[key])
    rows.append(w[FINAL_NORM_KEY])
    assert len(rows) == expected_norms

    norms = mx.stack([r.astype(mx.float32) for r in rows])  # [2L+1, hidden]
    assert tuple(norms.shape) == (expected_norms, hidden_size), norms.shape
    norms_bytes = np.asarray(norms, dtype="<f4").tobytes()
    norms_path = out / NORMS_FNAME
    norms_path.write_bytes(norms_bytes)
    norms_size = norms_path.stat().st_size
    print(
        f"   wrote {norms_path} ({norms_size:,} bytes, shape [{expected_norms}, {hidden_size}])"
    )
    print(
        f"   ordering: for i in 0..{n_layers - 1}: input_layernorm(i), post_attention_layernorm(i); then final model.norm.weight"
    )

    # --- (b2) Qwen3-only q_norm/k_norm per-head RMSNorm gains -----------------
    qk_rows = []
    qk_head_dim = None
    for i in range(n_layers):
        for suffix in QK_NORM_SUFFIXES:
            key = f"model.layers.{i}.self_attn.{suffix}"
            if key in w:
                qk_rows.append(w[key])
                qk_head_dim = int(w[key].shape[0])
            else:
                qk_rows = []
                break
        if not qk_rows:
            break
    qk_path = None
    qk_size = 0
    if qk_rows:
        qk = mx.stack([r.astype(mx.float32) for r in qk_rows])  # [2L, head_dim]
        qk_bytes = np.asarray(qk, dtype="<f4").tobytes()
        qk_path = out / QK_NORMS_FNAME
        qk_path.write_bytes(qk_bytes)
        qk_size = qk_path.stat().st_size
        print(
            f"   [qwen3] wrote {qk_path} ({qk_size:,} bytes, shape [{2 * n_layers}, {qk_head_dim}]) "
            f"row order: per layer q_norm(i), k_norm(i)"
        )
    else:
        print(
            "   no q_norm/k_norm in checkpoint (Qwen2.5-style) — skipping qk_norms.f32"
        )

    # --- (b3) per-projection BitLinear gains + biases (faithful forward) -----
    # Legacy (pre-deployed-forward) checkpoints carry these. Deployed-forward
    # checkpoints do not — those skip the file entirely, and the Rust runner
    # falls back to the weight-quant-only path.
    gains_path = None
    biases_path = None
    gains_size = biases_size = 0
    first_gain_key = f"model.layers.0.self_attn.q_proj.norm.weight"
    header = {}
    if first_gain_key in w:
        print("   [bitlinear] exporting gains.f32 + biases.f32 (faithful forward)...")
        gain_rows = []
        bias_rows = []
        # Biases are BF16 in the checkpoint; mx.load mis-reads them, so they
        # are read via the exact bit-shift instead.
        with open(args.checkpoint, "rb") as ck:
            _n = struct.unpack("<Q", ck.read(8))[0]
            header = json.loads(ck.read(_n))
        for i in range(n_layers):
            for proj in PROJ_ORDER:
                gkey = f"model.layers.{i}.{proj}.norm.weight"
                if gkey not in w:
                    sys.exit(f"error: '{gkey}' not found in legacy checkpoint")
                gain_rows.append(w[gkey].astype(mx.float32))
            for proj in BIAS_PROJS:
                bkey = f"model.layers.{i}.{proj}.bias"
                if bkey in header:
                    bias_rows.append(
                        mx.array(read_bf16_tensor(Path(args.checkpoint), header, bkey))
                    )
        gains = mx.concatenate(gain_rows, axis=0)
        gains_bytes = np.asarray(gains, dtype="<f4").tobytes()
        gains_path = out / GAINS_FNAME
        gains_path.write_bytes(gains_bytes)
        gains_size = gains_path.stat().st_size
        per_layer = 6 * hidden_size + intermediate_size
        print(
            f"   wrote {gains_path} ({gains_size:,} bytes, "
            f"{gains.shape[0]} floats = {n_layers} x ({6}*{hidden_size}+{intermediate_size}))"
        )
        if bias_rows:
            biases = mx.concatenate(bias_rows, axis=0)
            biases_bytes = np.asarray(biases, dtype="<f4").tobytes()
            biases_path = out / BIASES_FNAME
            biases_path.write_bytes(biases_bytes)
            biases_size = biases_path.stat().st_size
            print(
                f"   wrote {biases_path} ({biases_size:,} bytes, {biases.shape[0]} floats)"
            )
    else:
        print(
            "   [bitlinear] no per-projection gains in checkpoint "
            "(deployed-forward) — skipping gains.f32/biases.f32"
        )

    # --- hashes + index.json --------------------------------------------------
    print("4. hashing assets...")
    emb_sha = sha256_of(emb_path)
    norms_sha = sha256_of(norms_path)
    print(f"   {EMBED_FNAME} sha256 {emb_sha}")
    print(f"   {NORMS_FNAME} sha256 {norms_sha}")

    print("5. updating index.json (assets section)...")
    with open(idx_path, "r") as f:
        index = json.load(f)
    assets = {
        EMBED_FNAME: {
            "dtype": "f16",
            "shape": [vocab_size, hidden_size],
            "bytes": emb_size,
            "sha256": emb_sha,
            "source": f"{args.checkpoint} model.embed_tokens.weight (BF16->FP16)",
        },
        NORMS_FNAME: {
            "dtype": "f32",
            "shape": [expected_norms, hidden_size],
            "bytes": norms_size,
            "sha256": norms_sha,
            "source": f"{args.checkpoint} RMSNorm weights (BF16->FP32)",
        },
    }
    if qk_path is not None:
        qk_sha = sha256_of(qk_path)
        assets[QK_NORMS_FNAME] = {
            "dtype": "f32",
            "shape": [2 * n_layers, qk_head_dim],
            "bytes": qk_size,
            "sha256": qk_sha,
            "source": f"{args.checkpoint} Qwen3 q_norm/k_norm per-head RMSNorm (BF16->FP32)",
        }
    if gains_path is not None:
        assets[GAINS_FNAME] = {
            "dtype": "f32",
            "shape": [n_layers * (6 * hidden_size + intermediate_size)],
            "bytes": gains_size,
            "sha256": sha256_of(gains_path),
            "source": f"{args.checkpoint} per-projection BitLinear RMSNorm gains "
            f"(faithful forward, BF16->FP32)",
        }
    if biases_path is not None:
        assets[BIASES_FNAME] = {
            "dtype": "f32",
            "shape": [
                sum(
                    int(header[f"model.layers.{i}.{proj}.bias"]["shape"][0])
                    for i in range(n_layers)
                    for proj in BIAS_PROJS
                    if f"model.layers.{i}.{proj}.bias" in header
                )
            ],
            "bytes": biases_size,
            "sha256": sha256_of(biases_path),
            "source": f"{args.checkpoint} q/k/v projection biases (faithful forward, BF16->FP32)",
        }
    index["assets"] = assets
    # json.dump-style (no trailing newline), matching export_quantal_checkpoint.py
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"   updated {idx_path}")

    print("  done")


if __name__ == "__main__":
    main()
