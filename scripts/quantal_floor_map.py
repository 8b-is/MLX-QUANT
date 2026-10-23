#!/usr/bin/env python3
"""
quantal_floor_map.py — the fp16 (unquantized, no BitLinear) masked-CE floor
map of candidate base models on the quantal corpus.

This is the "floor map" gate for choosing the next training base
(Qwen3-1.7B vs Qwen3-4B): it tells us whether a bigger base's pretrained
knowledge already saturates the corpus (bases ≈ equal -> stop at 1.7B) or
whether there is headroom (4B meaningfully lower -> bet 4B).

Comparability note: this number is the *unquantized* floor on the SAME
masked-CE protocol as train_quantal_long.py (text field, dynamic per-batch
padding bucketed to *64, pad token 0 weighted out, stratified val 90,
seed 42). The old SOTA line is 0.5597 (fp16) and the ternary nightly best is
2.1469 (published in docs/kickoff-quantal-sota.md). The ratio
(quant-loss ≈ floor × ~1.3–3.8×) is what decides the base size: if the 4B
floor is not meaningfully below the 1.7B floor, the extra pretraining
knowledge is already saturated on this corpus and 1.7B stays the bet.

Eval-only: the base loads with plain mlx_lm weights (the checkpoint's native
dtype — Qwen3 ships bf16 — the same load call train_quantal*.py use, no
dtype override; mlx_lm 0.31.3 `load()` takes none). No BitLinear swap, no
training, no export.

Usage:
  python scripts/quantal_floor_map.py \
    --model Qwen/Qwen3-1.7B --data data/train_ultra_qwen3.jsonl \
    --max-len 256 --batch-size 8 --val-size 90
  python scripts/quantal_floor_map.py --model Qwen/Qwen3-4B  # the headroom probe

Runs on the rented GPU box (fork-mlx + mlx_lm in system python3, same as
nightly-quantal.sh: `PYTHONPATH="$PWD/python:$PWD/scripts"`).
"""

import argparse
import json
import os
import sys
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# 1. Data helpers (text field; dynamic per-batch padding, bucketed to *64)
# ---------------------------------------------------------------------------


def load_jsonl(path: str, max_samples: Optional[int] = None):
    samples = []
    with open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", obj.get("instruction", "")))
                if text:
                    samples.append(text)
            except json.JSONDecodeError:
                continue
    return samples


def make_batch(tokenizer, texts: list[str], max_len: int):
    """Dynamic per-batch padding bucketed to multiples of 64 (bounded JIT
    shape space — random shapes crash mlx-cuda's kernel cache)."""
    tokenized = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) < 2:
            continue
        tokenized.append(mx.array(ids[:max_len], dtype=mx.uint32))
    if not tokenized:
        return None
    max_t = min(max_len, max(x.shape[0] for x in tokenized))
    max_t = ((max_t + 63) // 64) * 64
    batch = []
    masks = []
    for x in tokenized:
        n = x.shape[0]
        padded = mx.pad(x, (0, max_t - n))
        mask = mx.zeros((max_t,), dtype=mx.bool_)
        mask[:n] = True
        batch.append(padded)
        masks.append(mask)
    inputs = mx.stack(batch)
    mask = mx.stack(masks)
    # fork-mlx (0.32.1.dev20260811) uses `axis=`, stock mlx uses `axes=` —
    # shim both so the same script runs on the Mac fork and on vast/mlx-cuda.
    try:
        targets = mx.roll(inputs, -1, axis=(1,))
    except TypeError:
        targets = mx.roll(inputs, -1, axes=(1,))
    targets[:, -1] = 0  # pad token 0, weighted out by mask
    return inputs, targets, mask


def ce_loss_masked(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Masked cross-entropy: pad positions weighted out, honest mean."""
    vocab = logits.shape[-1]
    lg = logits.reshape(-1, vocab)
    tg = targets.reshape(-1)
    mk = mask.reshape(-1).astype(mx.float32)
    return nn.losses.cross_entropy(lg, tg, reduction="mean", weights=mk)


# ---------------------------------------------------------------------------
# 2. Stratified val split (token-length buckets, deterministic)
# ---------------------------------------------------------------------------


def stratified_val_split(samples: list[str], tokenizer, val_size: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(samples)
    lens = np.array([len(tokenizer.encode(s)) for s in samples])
    # 5 length buckets
    buckets = np.clip((lens / max(1, lens.max()) * 4).astype(int), 0, 4)
    val_idx = []
    per_bucket = val_size // 5
    for b in range(5):
        cands = np.where(buckets == b)[0]
        if len(cands):
            rng.shuffle(cands)
            val_idx.extend(cands[:per_bucket].tolist())
    if len(val_idx) < val_size:
        rest = [i for i in range(n) if i not in set(val_idx)]
        rng.shuffle(rest)
        val_idx.extend(rest[: val_size - len(val_idx)])
    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


# ---------------------------------------------------------------------------
# 3. Batched masked eval (raw base — no BitLinear swap)
# ---------------------------------------------------------------------------


def evaluate(
    model, tokenizer, samples: list[str], max_len: int, batch_size: int = 4
) -> float:
    total = 0.0
    n = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        batch = make_batch(tokenizer, chunk, max_len)
        if batch is None:
            continue
        inputs, targets, mask = batch
        logits = model(inputs)
        loss = ce_loss_masked(logits, targets, mask)
        total += loss.item()
        n += 1
    return total / max(1, n)


# ---------------------------------------------------------------------------
# 4. main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="data/train_ultra_qwen3.jsonl")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-size", type=int, default=90)
    p.add_argument("--max-samples", type=int, default=None, help="cap on corpus reads")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print("  quantal floor map (fp16 masked-CE, unquantized, no BitLinear)")
    print(f"  model:     {args.model}")
    print(f"  data:      {args.data}")
    print(f"  max_len:   {args.max_len} (dynamic pad, x64 buckets)")
    print(
        f"  batch:     {args.batch_size}  val_size: {args.val_size}  seed: {args.seed}"
    )
    print(f"  metal:     {mx.metal.is_available()}")

    print("1. loading base model (plain mlx_lm weights)...")
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    from mlx.utils import tree_flatten

    tp = model.trainable_parameters()
    n_params = sum(v.nbytes for _, v in tree_flatten(tp)) / 1e6
    print(f"   loaded {n_params:.1f}M params")

    print("2. loading corpus...")
    samples = load_jsonl(args.data, max_samples=args.max_samples)
    if not samples:
        sys.exit("FATAL: no samples loaded")
    print(f"   {len(samples)} samples")

    print("3. stratified val split...")
    _, val_samples = stratified_val_split(samples, tokenizer, args.val_size, args.seed)
    print(f"   val {len(val_samples)} (stratified, seed {args.seed})")

    print("4. evaluating masked CE on the val split...")
    floor = evaluate(
        model, tokenizer, val_samples, args.max_len, batch_size=args.batch_size
    )
    print(f"  FP16 FLOOR ({args.model}): {floor:.4f}")
    print(f"  FLOOR_MASKED_CE={floor:.4f}")


if __name__ == "__main__":
    main()
