#!/usr/bin/env python3
"""
teacher_logits.py — Phase 1 of the distillation lane: one-time teacher-logits
cache for the quantal ternary brain (Qwen3-1.7B student, teacher
Qwen/Qwen3-32B-FP8, tokenizer byte-identical — vocab 151643, 0 missing, all
ids match, so per-position logits-KL is valid, see docs/distillation-spec.md).

For every sample: forward pass, keep the TOP-K logits per position (sparse —
1/2365 of full-vocab storage for top_k=64), write one npz per sample to
--out, plus a manifest.json that lets the training side (train_quantal_distill.py)
locate each sample's cache. The load-bearing alignment detail: each npz is
named by the sha1 of the sample's *text* (same text-field extraction as the
training side's load_jsonl), NOT by corpus position — the training loop permutes
samples every epoch, so position-based lookup would silently misalign.

Run once on the 80GB box (fp8-capable mlx + mlx_lm, see nightly-quantal.sh for
the box environment). THIS SCRIPT REFUSES TO RUN ON DEQUANTIZED WEIGHTS: the
teacher ships fp8 e4m3 and the KL target is only valid if the logits come from
the real fp8 model. If this mlx build has no fp8 dtype, or the load silently
dequantizes, the script exits loudly instead of caching wrong targets.

Usage:
  python scripts/teacher_logits.py \
    --teacher Qwen/Qwen3-32B-FP8 --data data/train_ultra_qwen3.jsonl \
    --out data/teacher_logits/ --max-len 256 --batch-size 8 \
    --max-samples 10000 --top-k 64
  (--max-samples 0 = whole corpus; probe 10k first per the spec budget.)

Deps: mlx + mlx_lm (fork build on the box), numpy. Standalone helpers copied
from train_quantal_long.py (same masked-CE protocol / dynamic pad x64).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# fp8 dtypes this mlx build can hold; empty on builds without fp8 support.
_FP8_DTYPES = tuple(
    getattr(mx, d)
    for d in ("float8_e4m3fn", "float8_e4m3fn_fast", "float8_e5m2", "float8_e5m2_fast")
    if hasattr(mx, d)
)


# ---------------------------------------------------------------------------
# 1. Data helpers (text field; dynamic per-batch padding, bucketed to *64)
#    — verbatim from train_quantal_long.py (the masked-CE protocol)
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


def sample_sha1(text: str) -> str:
    """The load-bearing alignment key: sha1 of the sample text (utf-8). Both
    teacher_logits.py and train_quantal_distill.py hash the SAME text-field
    string, so the cache lookup survives corpus re-ordering / re-splitting."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Top-k extraction (sparse per-position logits)
# ---------------------------------------------------------------------------


def topk_ids_logits(logits: mx.array, top_k: int):
    """Top-k token ids + logit values along the vocab axis.

    `mx.topk` in this mlx build returns VALUES ONLY (no ids), so ids come from
    argpartition on -logits (the k-th smallest of -logits = k-th largest of
    logits) + take_along_axis to gather the values. The result is unsorted
    within k — fine: the KL in train_quantal_distill is invariant to the order
    of the id set, teacher and student both gather at the same ids.
    """
    top_ids = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
    top_logits = mx.take_along_axis(logits, top_ids, axis=-1)
    return top_ids, top_logits


# ---------------------------------------------------------------------------
# 3. fp8 guard (fail loudly, never cache dequantized teacher logits)
# ---------------------------------------------------------------------------


def assert_fp8_teacher(model, teacher: str) -> None:
    from mlx.utils import tree_flatten

    leaves = [v for _, v in tree_flatten(model.parameters())]
    fp8_hits = [v for v in leaves if v.dtype in _FP8_DTYPES]
    # Only an FP8 teacher (tensors actually in fp8 e4m3/e5m2) needs an
    # fp8-capable mlx build. A bf16/fp32 teacher (e.g. Qwen3-14B, used when the
    # FP8 32B fails the weight_scale_inv schema) loads fine on any build —
    # requiring fp8 dtypes for it would wrongly abort a valid teacher.
    if not fp8_hits:
        print("   teacher is not fp8 — no fp8 dtype requirement")
        return
    if not _FP8_DTYPES:
        sys.exit(
            "FATAL: this mlx build has NO fp8 dtypes "
            f"({mx.__version__}) — cannot hold {teacher} natively. "
            "The teacher ships fp8 e4m3 weights; silently running on dequantized "
            "weights would corrupt the KL targets. The teacher needs an "
            "fp8-capable mlx build (the box's mlx-cuda fp8 build — see "
            "docs/distillation-spec.md Phase 1)."
        )
    print(f"   fp8 ok: {len(fp8_hits)} leaves in {_FP8_DTYPES}")


def weight_dtype_note(model) -> str:
    from mlx.utils import tree_flatten

    counts = {}
    for _, v in tree_flatten(model.parameters()):
        counts[str(v.dtype)] = counts.get(str(v.dtype), 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# 4. main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen3-32B-FP8")
    p.add_argument("--data", default="data/train_ultra_qwen3.jsonl")
    p.add_argument("--out", default="data/teacher_logits/")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-samples", type=int, default=10000, help="probe default; 0 = whole corpus"
    )
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if "MLX_CUDA_GRAPH_CACHE_SIZE" not in os.environ:
        os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "2000"
        print("  note: MLX_CUDA_GRAPH_CACHE_SIZE unset — defaulted to 2000")
    else:
        print(
            f"  note: MLX_CUDA_GRAPH_CACHE_SIZE={os.environ['MLX_CUDA_GRAPH_CACHE_SIZE']} (respected)"
        )

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("  teacher logits cache (distillation Phase 1)")
    print(f"  teacher:   {args.teacher}")
    print(f"  data:      {args.data}")
    print(f"  out:       {args.out}")
    print(f"  max_len:   {args.max_len}  batch: {args.batch_size}  top_k: {args.top_k}")
    print(f"  max_samples: {args.max_samples if args.max_samples else 'ALL'}")
    print(f"  mlx:       {mx.__version__}  metal: {mx.metal.is_available()}")

    print("1. loading teacher...")
    from mlx_lm.utils import load as mlx_load

    try:
        model, tokenizer = mlx_load(args.teacher)
    except Exception as e:
        sys.exit(
            f"FATAL: mlx_load({args.teacher}) failed: {e!r}. If this mlx build "
            "lacks fp8 (e4m3) support the safetensors load fails here — the "
            "teacher needs an fp8-capable mlx build (see docs/distillation-spec.md "
            "Phase 1)."
        )
    assert_fp8_teacher(model, args.teacher)
    print(f"   loaded {args.teacher} — weight dtypes: {weight_dtype_note(model)}")

    tokenizer_id = getattr(tokenizer, "vocab_size", None)
    if tokenizer_id is None:
        tokenizer_id = (
            len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else "unknown"
        )
    print(f"   tokenizer vocab_size={tokenizer_id}")

    print("2. loading corpus...")
    samples = load_jsonl(
        args.data, max_samples=None if args.max_samples == 0 else args.max_samples
    )
    if not samples:
        sys.exit("FATAL: no samples loaded")
    print(f"   {len(samples)} samples")

    print("3. caching top-k logits...")
    n_written = 0
    n_batches = max(1, (len(samples) + args.batch_size - 1) // args.batch_size)
    hashes = []
    t0 = time.time()
    for i in range(0, len(samples), args.batch_size):
        chunk = samples[i : i + args.batch_size]
        # make_batch drops <2-token texts; pre-filter the SAME way so batch rows
        # map 1:1 to the texts we name npz files after (hash-keyed anyway, but
        # this keeps the row correspondence exact).
        kept = [t for t in chunk if len(tokenizer.encode(t)) >= 2]
        batch = make_batch(tokenizer, kept, args.max_len)
        if batch is None:
            continue
        inputs, _, mask = batch
        logits = model(inputs)
        top_ids, top_logits = topk_ids_logits(logits, args.top_k)
        for j, text in enumerate(kept):
            n = int(np.asarray(mask[j].tolist()).sum())
            h = sample_sha1(text)
            np.savez(
                os.path.join(args.out, f"{h}.npz"),
                ids=np.asarray(inputs[j, :n].tolist(), dtype=np.uint32),
                top_ids=np.asarray(top_ids[j, :n].tolist(), dtype=np.uint32),
                top_logits=np.asarray(top_logits[j, :n].tolist(), dtype=np.float32),
                mask=np.asarray(mask[j, :n].tolist(), dtype=bool),
            )
            hashes.append(h)
            n_written += 1
        if (i // args.batch_size) % 10 == 0:
            print(
                f"   batch {i // args.batch_size + 1}/{n_batches} | "
                f"{n_written} samples | {time.time() - t0:.0f}s",
                flush=True,
            )
    manifest = {
        "teacher": args.teacher,
        "top_k": args.top_k,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "tokenizer_id": tokenizer_id,
        "sample_count": n_written,
        "generated_by": "scripts/teacher_logits.py",
        "files": [{"file": f"{h}.npz", "sha1": h} for h in hashes],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"   manifest -> {os.path.join(args.out, 'manifest.json')} ({n_written} files)"
    )
    print(f"  done in {time.time() - t0:.0f}s")
    print(f"  TEACHER_LOGITS_DONE {n_written} top_k={args.top_k}")


if __name__ == "__main__":
    main()
