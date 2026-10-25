#!/usr/bin/env python3
"""
train_quantal_long.py — long quantal continued-train (ULTRA / LOVEGOD MODE).

Reconstructed from the baszataska-night protocol (train_quantal_long.py that
ran on vast): masked CE, deployed-forward QAT, early stop, real stratified val.
Key lessons baked in:
  - grad clip OFF (mx.clip over the grad tree is flaky on mlx-cuda)
  - value_and_grad (single graph, no double loss_fn/model.update)
  - mx-aware LR schedule (pure-Python cosine returns float past warmup ->
    crashes apply_single's .astype)
  - deployed forward: weight-quant-only BitLinear (per-projection RMSNorm +
    activation_quant skipped) so training == inference in the Rust runner.
    Default since the deployed-forward fix; --no-deployed-forward restores
    the faithful b1.58 forward for legacy checkpoint resumes.

Usage:
  python scripts/train_quantal_long.py \
    --model Qwen/Qwen2.5-0.5B --data train.jsonl \
    --batch-size 12 --max-len 512 --epochs 40 --val-size 90 \
    --outdir ckpts --curve curve.jsonl [--resume ckpts/quantal-long-best.safetensors]

Qwen3 (quant-ultra round): the --model flag is architecture-agnostic —
  python scripts/train_quantal_long.py \
    --model Qwen/Qwen3-1.7B --data train_ultra_qwen3.jsonl \
    --batch-size 12 --max-len 512 --epochs 40 --val-size 90 --outdir ckpts \
    --curve curve.jsonl
  (28 layers x 7 projections = 196 BitLinears; vocab 151936 unchanged;
  tied embeddings -> no lm_head swap. mx.roll `axes`/`axis` shim applied.)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_map

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx.nn.layers.bitlinear import BitLinear  # noqa: E402
from train_quantal import replace_linear_with_bitlinear  # noqa: E402

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
# 2. LR schedule (mx-aware: always returns mx.array)
# ---------------------------------------------------------------------------


def make_schedule(lr_init: float, lr_end: float, warmup_frac: float, total_steps: int):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def schedule(step):
        step = mx.array(step)
        in_warmup = step < warmup_steps
        warm_lr = lr_init * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = mx.maximum(mx.minimum(t, 1.0), 0.0)
        cos_lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + mx.cos(mx.pi * t))
        return mx.where(in_warmup, warm_lr, cos_lr)

    return schedule


# ---------------------------------------------------------------------------
# 3. Stratified val split (token-length buckets, deterministic)
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
# 4. Batched masked eval
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
# 5. main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", required=True)
    p.add_argument("--lr-init", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--val-size", type=int, default=90)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument(
        "--grad-clip", type=float, default=0.0, help="0 = off (flaky on mlx-cuda)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="ckpts")
    p.add_argument("--curve", default="curve.jsonl")
    p.add_argument("--resume", default=None, help="safetensors to resume from")
    p.add_argument(
        "--deployed-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight-quant-only forward == Rust runner (default). "
        "Pass --no-deployed-forward to rebuild the faithful "
        "per-projection-RMSNorm BitLinear forward for a legacy "
        "checkpoint.",
    )
    args = p.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("  quantal training (oracle protocol, deployed forward, ULTRA MODE)")
    print(f"  model:     {args.model}")
    print(f"  data:      {args.data}")
    print(f"  resume:    {args.resume or 'fresh from base'}")
    print(
        f"  lr:        {args.lr_init} -> cosine -> {args.lr_end}, warmup {args.warmup_frac}"
    )
    print(f"  batch:     {args.batch_size}  max_len: {args.max_len} (dynamic pad)")
    print(f"  epochs:    {args.epochs} (cap)")
    print(
        f"  val_size:  {args.val_size}  patience: {args.patience}  min_delta: {args.min_delta}"
    )
    print(f"  wd:        {args.weight_decay}  grad_clip: {args.grad_clip}")
    print(f"  metal:     {mx.metal.is_available()}")

    print("1. loading base model...")
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    from mlx.utils import tree_flatten

    tp = model.trainable_parameters()
    n_params = sum(v.nbytes for _, v in tree_flatten(tp)) / 1e6
    print(f"   loaded {n_params:.1f}M params")

    print("2. swapping Linear -> BitLinear (QAT, deployed forward)...")
    n_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    model = replace_linear_with_bitlinear(model, deployed_forward=args.deployed_forward)
    n_after = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(f"   replaced {n_before} Linear -> {n_after} BitLinear")

    if args.resume:
        print(f"3. loading checkpoint weights <- {args.resume}")
        model.load_weights(args.resume)

    print("4. loading data...")
    samples = load_jsonl(args.data)
    if not samples:
        sys.exit("FATAL: no samples loaded")
    train_samples, val_samples = stratified_val_split(
        samples, tokenizer, args.val_size, args.seed
    )
    print(f"   {len(samples)} samples")
    print(f"   train {len(train_samples)} / val {len(val_samples)} (stratified)")

    n_batches = max(1, len(train_samples) // args.batch_size)
    total_steps = n_batches * args.epochs
    schedule = make_schedule(args.lr_init, args.lr_end, args.warmup_frac, total_steps)
    optimizer = opt.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)
    print(
        f"5. optimizer AdamW wd={args.weight_decay}, {n_batches} batches/epoch, {total_steps} total steps"
    )

    print("6. training...")
    best_val = float("inf")
    bad_epochs = 0
    global_step = 0

    def loss_fn(params):
        model.update(params)
        logits = model(inputs)
        return ce_loss_masked(logits, targets, mask)

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        steps_done = 0
        idxs = np.random.permutation(len(train_samples))

        for batch_idx in range(0, len(idxs), args.batch_size):
            batch_idxs = idxs[batch_idx : batch_idx + args.batch_size]
            batch = make_batch(
                tokenizer, [train_samples[i] for i in batch_idxs], args.max_len
            )
            if batch is None:
                continue
            inputs, targets, mask = batch

            params = model.trainable_parameters()
            loss, grads = mx.value_and_grad(loss_fn)(params)
            if args.grad_clip and args.grad_clip > 0:
                grads = tree_map(
                    lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grads
                )
            optimizer.update(model, grads)

            lv = loss.item()
            epoch_loss += lv
            steps_done += 1
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"   step {global_step}/{total_steps} | loss {lv:.4f} | lr {float(schedule(global_step)):.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, steps_done)
        elapsed = time.time() - t0
        sps = steps_done / max(1.0, elapsed)

        # Eval with the same batch size as training: the CUDA graph cache is
        # warmed on training shapes, so a smaller eval batch would JIT-build new
        # graph shapes under memory pressure and cublasLtMatmul can fail (code 7).
        val_loss = evaluate(
            model, tokenizer, val_samples, args.max_len, batch_size=args.batch_size
        )
        lr_now = float(schedule(global_step))

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": lr_now,
            "elapsed_s": round(elapsed, 1),
            "steps_per_sec": round(sps, 2),
            "best_val": round(best_val, 4),
            "global_step": global_step,
        }
        with open(args.curve, "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"  epoch {epoch + 1}/{args.epochs} | loss {avg_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s ({sps:.2f} steps/s)",
            flush=True,
        )

        # save every-5 and best
        if (epoch + 1) % 5 == 0:
            pth = os.path.join(
                args.outdir, f"quantal-long-epoch-{epoch + 1}.safetensors"
            )
            model.save_weights(pth)
            print(f"   [save] weights -> {pth}", flush=True)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad_epochs = 0
            pth = os.path.join(args.outdir, "quantal-long-best.safetensors")
            model.save_weights(pth)
            print(f"   [save] weights -> {pth}", flush=True)
            print(f"   * new best val {best_val:.4f}", flush=True)
        else:
            bad_epochs += 1
            print(
                f"   val not improved (bad_epochs={bad_epochs}/{args.patience}, best {best_val:.4f})",
                flush=True,
            )
            if bad_epochs >= args.patience:
                print("   early stop", flush=True)
                break

    pth = os.path.join(args.outdir, "quantal-long-final.safetensors")
    model.save_weights(pth)
    print(f"  done. best masked-val {best_val:.4f}. final -> {pth}", flush=True)
    print(f"  FINAL_MASKED_VAL={best_val:.4f}")


if __name__ == "__main__":
    main()
