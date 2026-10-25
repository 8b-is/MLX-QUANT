#!/usr/bin/env python3
"""
quantal_golden_logits.py — golden-logits reference for the native ternary runner.

Reconstructs the quantal (BitNet b1.58) model's forward from the SAME ayeOS
codes/scales + embeddings.f16 + norms.f32 the Rust runner consumes, and
captures final-layer logits for the two fixed gate prompts.

Two variants are computed:
  * vanilla  — plain Qwen2.5 with ternary weights (what the Rust runner implements)
  * bitlinear — the trained checkpoint's faithful BitLinear forward: per-projection
               activation RMSNorm + int8 activation_quant + q/k/v biases
               (weights = dequant(codes, scales); extras loaded from the
               checkpoint safetensors — NOT exported to ayeOS)

Usage:
  python scripts/quantal_golden_logits.py \
      --model-dir demos/quantal \
      [--checkpoint demos/quantal/quantal_model.safetensors] \
      [--out ref_logits.json]

The Rust side (`cargo run -p ternary --example quantal_logits`) emits the
runner's logits; `quantal_compare_logits.py` does the acceptance comparison.
Gate prompts MUST match crates/ternary/examples/quantal_logits.rs.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

# --- Qwen2.5-0.5B architecture (must mirror crates/ternary/src/model.rs) ---
HIDDEN = 896
LAYERS = 24
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 4864
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6
GROUP = Q_HEADS // KV_HEADS  # 7
INV_SQRT = 1.0 / np.sqrt(HEAD_DIM)

GATE_PROMPTS = [
    (
        "You are a helpful assistant.",
        "What is the capital of France? Answer in one word.",
    ),
    (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "Explain the concept of recursion in one short paragraph.",
    ),
]


# ---------------------------------------------------------------- dequant ---


def dequant_matrix(codes, scales, n, k):
    """ayeOS: value = (code − 1) · scale, 2-bit LSB-first, row-major over K."""
    c = np.asarray(codes, dtype=np.uint32).reshape(n, k // 16)
    lanes = np.stack([(c >> (2 * i)) & 3 for i in range(16)], axis=-1)  # (N, K/16, 16)
    vals = lanes.reshape(n, k).astype(np.float32) - 1.0
    s = np.repeat(np.asarray(scales, dtype=np.float32).reshape(n, k // 64), 64, axis=1)
    return mx.array(vals * s)


def load_ayes(capsule):
    """Load mNNN.json matrices keyed by name (manifest order from index.json)."""
    index = json.loads((capsule / "index.json").read_text())
    mats = {}
    for entry in index["matrices"]:
        with open(capsule / entry["file"]) as f:
            raw = json.load(f)
        mats[raw["name"]] = dequant_matrix(
            raw["codes"], raw["scales"], raw["dim"], raw["in_features"]
        )
    return mats


def load_embeddings(capsule):
    data = np.fromfile(capsule / "embeddings.f16", dtype=np.float16)
    return mx.array(data.astype(np.float32).reshape(-1, HIDDEN))  # [V, H]


def load_norms(capsule):
    return np.fromfile(capsule / "norms.f32", dtype=np.float32).reshape(-1, HIDDEN)


# ------------------------------------------------------------------ blocks ---


def rmsnorm(x, w, eps=RMS_EPS):
    return x * w / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def rope_freqs():
    return np.array(
        [1.0 / ROPE_THETA ** (2.0 * i / HEAD_DIM) for i in range(HEAD_DIM // 2)],
        dtype=np.float32,
    )


FREQS = rope_freqs()


def rope_apply(x, pos):
    """HF-style rotate_half. x: (..., D); returns rotated copy at absolute `pos`."""
    angle = pos * FREQS  # (D/2,)
    c = np.concatenate([np.cos(angle), np.cos(angle)], axis=-1)  # (D,)
    s = np.concatenate([np.sin(angle), np.sin(angle)], axis=-1)
    half = HEAD_DIM // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * mx.array(c) + rotated * mx.array(s)


def silu(x):
    return x * mx.sigmoid(x)


def attention(q, k, v):
    """Causal GQA over a full sequence. q (S,14,64), k/v (S,2,64) already roped."""
    s = q.shape[0]
    score_rows = []
    for qh in range(Q_HEADS):
        g = qh // GROUP
        score_rows.append((q[:, qh, :] @ k[:, g, :].T) * INV_SQRT)
    scores = mx.stack(score_rows, axis=1)  # (S, 14, S)
    mask = mx.array(np.triu(np.ones((s, s), dtype=np.float32), k=1)).reshape(s, 1, s)
    scores = mx.where(mask, -1e30, scores)
    probs = mx.softmax(scores, axis=-1)
    out_rows = []
    for qh in range(Q_HEADS):
        g = qh // GROUP
        out_rows.append(probs[:, qh, :] @ v[:, g, :])  # (S, 64)
    return mx.stack(out_rows, axis=1).reshape(s, Q_HEADS * HEAD_DIM)


# ------------------------------------------------------------- projections ---


def act_quant(x):
    """int8 simulated activation quantization (BitLinear training forward)."""
    scale = 127.0 / mx.clip(mx.abs(x).max(axis=-1, keepdims=True), 1e-5, None)
    return mx.clip(mx.round(x * scale), -128, 127) / scale


def linear(x, w, bitlinear=False, norm_w=None, bias=None):
    """w: [out, in]. bitlinear → activation_quant(rmsnorm(x)) @ w.T + bias."""
    if bitlinear:
        x = act_quant(rmsnorm(x, mx.array(norm_w)))
        y = x @ w.T
        if bias is not None:
            y = y + mx.array(bias)
        return y
    return x @ w.T


# ----------------------------------------------------------------- forward ---


def _norm_w(extras, layer, key):
    """norm weight for a projection when extras are active."""
    return extras[layer][key] if extras is not None else None


def _bias(extras, layer, key):
    return extras[layer][key] if extras is not None else None


def forward(tokens, mats, emb, norms, bitlinear=False, extras=None):
    """Full prefill; returns last-token logits [V]."""
    tok = mx.array(np.asarray(tokens, dtype=np.uint32))
    hidden = emb[tok]  # (S, H)

    for layer in range(LAYERS):
        in_norm = norms[2 * layer]
        post_norm = norms[2 * layer + 1]
        name = f"model.layers.{layer}"

        x = rmsnorm(hidden, mx.array(in_norm))
        q = linear(
            x,
            mats[f"{name}.self_attn.q_proj"],
            bitlinear,
            _norm_w(extras, layer, "q.norm"),
            _bias(extras, layer, "q.bias"),
        )
        k = linear(
            x,
            mats[f"{name}.self_attn.k_proj"],
            bitlinear,
            _norm_w(extras, layer, "k.norm"),
            _bias(extras, layer, "k.bias"),
        )
        v = linear(
            x,
            mats[f"{name}.self_attn.v_proj"],
            bitlinear,
            _norm_w(extras, layer, "v.norm"),
            _bias(extras, layer, "v.bias"),
        )

        s = q.shape[0]
        q = q.reshape(s, Q_HEADS, HEAD_DIM)
        k = k.reshape(s, KV_HEADS, HEAD_DIM)
        v = v.reshape(s, KV_HEADS, HEAD_DIM)
        q = mx.stack([rope_apply(q[i], i) for i in range(s)], axis=0)
        k = mx.stack([rope_apply(k[i], i) for i in range(s)], axis=0)

        attn = attention(q, k, v)
        o = linear(
            attn,
            mats[f"{name}.self_attn.o_proj"],
            bitlinear,
            _norm_w(extras, layer, "o.norm"),
            None,
        )
        hidden = hidden + o

        x = rmsnorm(hidden, mx.array(post_norm))
        up = linear(
            x,
            mats[f"{name}.mlp.up_proj"],
            bitlinear,
            _norm_w(extras, layer, "up.norm"),
            None,
        )
        gate = linear(
            x,
            mats[f"{name}.mlp.gate_proj"],
            bitlinear,
            _norm_w(extras, layer, "gate.norm"),
            None,
        )
        act = silu(gate) * up
        down = linear(
            act,
            mats[f"{name}.mlp.down_proj"],
            bitlinear,
            _norm_w(extras, layer, "down.norm"),
            None,
        )
        hidden = hidden + down

    hidden = rmsnorm(hidden, mx.array(norms[2 * LAYERS]))
    return (hidden[-1] @ emb.T).astype(mx.float32)


# ------------------------------------------------------------------- extras ---


def _read_bf16_tensor(path, header, key):
    """Read a single BF16 tensor from a safetensors file (header parse).

    ``data_offsets`` in safetensors are relative to the START OF THE DATA
    SECTION (after the 8-byte header-length prefix + header JSON), not the
    file start. Reading from the raw offset instead lands 8+header_len bytes
    early and returns the neighbouring tensor's bytes.
    """
    import struct as _struct

    info = header[key]
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        hdr_len = _struct.unpack("<Q", f.read(8))[0]
        data_start = 8 + hdr_len
        f.seek(data_start + start)
        raw = f.read(end - start)
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    # BF16 -> F32: shift the 16-bit value into the top half of an f32 bit
    # pattern, then REINTERPRET as float32. Without the `.view(np.float32)`
    # the shifted bits are a huge uint32 integer (the numeric value of the
    # bit pattern, ~1e9), which corrupts every biased projection.
    return (bits << 16).view(np.float32)


def load_bitlinear_extras(checkpoint):
    """Per-layer {proj}.norm.weight (F32) + q/k/v biases (BF16) from the
    trained checkpoint. Norm weights load via safetensors' mlx backend; the
    biases are BF16 (which that backend rejects) so they are read by hand."""
    import json
    import struct

    from safetensors import safe_open

    path = str(checkpoint)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))

    extra = {}
    with safe_open(path, framework="mlx") as f:
        for layer in range(LAYERS):
            name = f"model.layers.{layer}"
            ex = {}
            for proj, bias_key in (
                ("q", "self_attn.q_proj"),
                ("k", "self_attn.k_proj"),
                ("v", "self_attn.v_proj"),
                ("o", "self_attn.o_proj"),
                ("up", "mlp.up_proj"),
                ("gate", "mlp.gate_proj"),
                ("down", "mlp.down_proj"),
            ):
                key = f"{name}.{bias_key}.norm.weight"
                ex[proj + ".norm"] = f.get_slice(key)[:].astype(mx.float32)
                bkey = f"{name}.{bias_key}.bias"
                if bkey in header:
                    ex[proj + ".bias"] = mx.array(_read_bf16_tensor(path, header, bkey))
            extra[layer] = ex
    return extra


# --------------------------------------------------------------------- main ---


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="quantal_ref_logits.json")
    p.add_argument(
        "--vanilla-only",
        action="store_true",
        help="skip the bitlinear variant (no checkpoint needed)",
    )
    args = p.parse_args()

    capsule = Path(args.model_dir)
    print("1. loading tokenizer + matrices + embeddings + norms...")
    tok = Tokenizer.from_file(str(capsule / "tokenizer.json"))
    mats = load_ayes(capsule)
    emb = load_embeddings(capsule)
    norms = load_norms(capsule)
    print(f"   {len(mats)} matrices, embeddings {emb.shape}, norms {norms.shape}")

    extras = None
    if not args.vanilla_only:
        ckpt = (
            Path(args.checkpoint)
            if args.checkpoint
            else capsule / "quantal_model.safetensors"
        )
        print(f"2. loading BitLinear extras from {ckpt}...")
        extras = load_bitlinear_extras(ckpt)

    results = []
    for i, (system, user) in enumerate(GATE_PROMPTS):
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = tok.encode(prompt, add_special_tokens=False).ids
        print(f"3. prompt {i + 1}: {len(ids)} tokens")
        logits_v = forward(ids, mats, emb, norms, bitlinear=False)
        top = mx.argmax(logits_v)
        print(f"   vanilla  argmax={top.item()} ({tok.decode([top.item()])!r})")
        row = {
            "prompt": i + 1,
            "system": system,
            "user": user,
            "n_tokens": len(ids),
            "logits_vanilla": [float(x) for x in logits_v.tolist()],
        }
        if extras is not None:
            logits_b = forward(ids, mats, emb, norms, bitlinear=True, extras=extras)
            topb = mx.argmax(logits_b)
            print(f"   bitlinear argmax={topb.item()} ({tok.decode([topb.item()])!r})")
            row["logits_bitlinear"] = [float(x) for x in logits_b.tolist()]
        results.append(row)

    with open(args.out, "w") as f:
        json.dump({"model_dir": str(capsule.resolve()), "prompts": results}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
