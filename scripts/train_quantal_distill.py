#!/usr/bin/env python3
"""
train_quantal_distill.py — quantal continued-train with teacher-KL distillation
(distillation Phase 2, docs/distillation-spec.md).

Identical to train_quantal_long.py (masked CE, deployed-forward QAT, early
stop, real stratified val) EXCEPT the loss gains a masked-KL term against the
teacher-logits cache produced by scripts/teacher_logits.py (Phase 1):

    loss = (1 - λ)·masked_CE + λ·masked_KL,   λ = --distill-weight (fixed 0.5)

masked_KL: per position, gather the STUDENT logits at the teacher's top-k ids
(from the cache), softmax both sides over that top-k, sum p_t·log(p_t/p_s),
masked over valid tokens (pad token 0 weighted out — same mask as the CE).

Parity invariant: the student forward stays the deployed-forward thresholded
ternary BitLinear (`replace_linear_with_bitlinear(model, deployed_forward=True)`)
— only the LOSS changes, the forward is untouched, so the Rust runner still
reads the exported codes+scales unchanged.

Alignment (load-bearing): the cache is keyed by the sha1 of each sample's text
(both scripts hash the same load_jsonl text field), NOT by corpus position —
the training loop permutes samples every epoch, so the hash map is the only
safe lookup. Samples missing from the cache contribute KL 0 (weighted out,
never a crash) and are counted per epoch.

`--distill-weight 0` runs the EXACT train_quantal_long behavior (CE only) so we
can A/B against the CE-only run on the same stratified val 90 split.

Usage:
  python scripts/train_quantal_distill.py \
    --model Qwen/Qwen3-1.7B --data data/train_ultra_qwen3.jsonl \
    --batch-size 12 --max-len 256 --epochs 40 --val-size 90 \
    --distill-dir data/teacher_logits/ --distill-weight 0.5 --top-k 64 \
    --outdir ckpts-distill --curve curve-distill.jsonl
  # CE-only baseline (identical to train_quantal_long):
  python scripts/train_quantal_distill.py \
    --model Qwen/Qwen3-1.7B --data data/train_ultra_qwen3.jsonl \
    --batch-size 12 --max-len 256 --epochs 40 --val-size 90 \
    --distill-weight 0 --outdir ckpts --curve curve.jsonl
"""

import argparse
import hashlib
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


def sample_sha1(text: str) -> str:
    """Load-bearing alignment key: sha1 of the sample text (utf-8), computed by
    BOTH teacher_logits.py and this script on the SAME load_jsonl text-field
    string. The teacher cache is looked up by this hash, never by corpus
    position (training permutes samples every epoch)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _build_batch(tokenizer, texts: list[str], max_len: int):
    """Tokenize + filter (<2 tokens dropped — same rule as make_batch) + x64
    bucket + pad + roll shim. Returns (inputs, targets, mask, kept_texts) where
    kept_texts are the rows that actually landed in the batch (cache rows must
    align 1:1 with them)."""
    tokenized = []
    kept = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) < 2:
            continue
        kept.append(t)
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
    return inputs, targets, mask, kept


def make_batch(tokenizer, texts: list[str], max_len: int):
    """Dynamic per-batch padding bucketed to multiples of 64 (bounded JIT
    shape space — random shapes crash mlx-cuda's kernel cache). Identical
    behavior to train_quantal_long.make_batch."""
    b = _build_batch(tokenizer, texts, max_len)
    if b is None:
        return None
    inputs, targets, mask, _ = b
    return inputs, targets, mask


def make_batch_aligned(
    tokenizer, texts: list[str], max_len: int, cache_index, top_k: int
):
    """make_batch + teacher-cache rows aligned to the SAME batch rows (the
    <2-token filter is applied inside _build_batch, so rows map 1:1 to the kept
    texts). Returns (inputs, targets, mask, cache_batch|None, n_missing)."""
    b = _build_batch(tokenizer, texts, max_len)
    if b is None:
        return None
    inputs, targets, mask, kept = b
    cache_batch = None
    n_missing = 0
    if cache_index is not None:
        cache_batch, n_missing = build_cache_batch(
            kept, inputs.shape[1], cache_index, top_k
        )
    return inputs, targets, mask, cache_batch, n_missing


def ce_loss_masked(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Masked cross-entropy: pad positions weighted out, honest mean."""
    vocab = logits.shape[-1]
    lg = logits.reshape(-1, vocab)
    tg = targets.reshape(-1)
    mk = mask.reshape(-1).astype(mx.float32)
    return nn.losses.cross_entropy(lg, tg, reduction="mean", weights=mk)


# ---------------------------------------------------------------------------
# 1b. Teacher-cache loading + masked KL
# ---------------------------------------------------------------------------


def load_cache_index(distill_dir: str, top_k: int):
    """Build {sha1: npz path} from the Phase-1 manifest; validate top_k.
    Returns (index, manifest)."""
    manifest_path = os.path.join(distill_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(
            f"FATAL: {manifest_path} not found — run scripts/teacher_logits.py "
            "first (distillation Phase 1)."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("top_k") != top_k:
        sys.exit(
            f"FATAL: cache top_k={manifest.get('top_k')} != --top-k {top_k}. "
            "The cache was generated with a different top-k — regenerate it or "
            f"pass --top-k {manifest.get('top_k')}."
        )
    index = {}
    for f in manifest["files"]:
        index[f["sha1"]] = os.path.join(distill_dir, f["file"])
    return index, manifest


def build_cache_batch(texts: list[str], max_t: int, cache_index, top_k: int):
    """Stack per-sample teacher caches into (B, max_t, K) rows aligned to the
    batch produced for the SAME texts. A sample without a cache file gets a
    zero row with mask all-False — its KL term is dropped (weighted 0), never a
    crash. Caches are truncated to max_t when longer (training max-len may be
    <= the cache max-len). Returns ((top_ids, top_logits, kl_mask), n_missing).
    """
    rows = []
    n_missing = 0
    for t in texts:
        path = cache_index.get(sample_sha1(t))
        if path is None:
            n_missing += 1
            rows.append(
                (
                    mx.zeros((max_t, top_k), dtype=mx.int32),
                    mx.zeros((max_t, top_k), dtype=mx.float32),
                    mx.zeros((max_t,), dtype=mx.bool_),
                )
            )
            continue
        with np.load(path) as z:
            c_ids = np.asarray(z["top_ids"][:max_t], dtype=np.int32)
            c_lg = np.asarray(z["top_logits"][:max_t], dtype=np.float32)
            c_mk = np.asarray(z["mask"][:max_t], dtype=bool)
        n = c_mk.shape[0]
        k = c_ids.shape[1]
        if k != top_k:
            sys.exit(f"FATAL: cache K={k} != --top-k {top_k} in {path}")
        rows.append(
            (
                mx.pad(mx.array(c_ids), ((0, max_t - n), (0, 0))),
                mx.pad(mx.array(c_lg), ((0, max_t - n), (0, 0))),
                mx.pad(mx.array(c_mk), ((0, max_t - n),)),
            )
        )
    if not rows:
        return None, n_missing
    top_ids = mx.stack([r[0] for r in rows])
    top_logits = mx.stack([r[1] for r in rows])
    kl_mask = mx.stack([r[2] for r in rows])
    return (top_ids, top_logits, kl_mask), n_missing


def kl_loss_distill(
    student_logits: mx.array,
    teacher_top_ids: mx.array,
    teacher_top_logits: mx.array,
    mask: mx.array,
) -> mx.array:
    """Per-position KL over the teacher's top-k ids: sum_t p_t·log(p_t/p_s),
    masked over valid tokens (pad positions weighted out — same masking
    philosophy as ce_loss_masked). log-softmax form for stability. The teacher
    and student share the vocab, so ids are clamped defensively (should never
    fire)."""
    ids = mx.clip(teacher_top_ids, 0, student_logits.shape[-1] - 1).astype(mx.int32)
    s_top = mx.take_along_axis(student_logits, ids, axis=-1)  # (B, T, K)
    t_lse = teacher_top_logits - mx.logsumexp(
        teacher_top_logits, axis=-1, keepdims=True
    )
    s_lse = s_top - mx.logsumexp(s_top, axis=-1, keepdims=True)
    p_t = mx.exp(t_lse)
    kl = mx.sum(p_t * (t_lse - s_lse), axis=-1)  # (B, T) per-position KL
    mk = mask.astype(mx.float32)
    return mx.sum(kl * mk) / mx.maximum(mx.sum(mk), 1e-6)


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
# 4. Batched masked eval (CE-only, and CE+KL for the distill lane)
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


def evaluate_distill(
    model, tokenizer, samples, max_len, batch_size, cache_index, top_k
):
    """evaluate + masked-KL against the teacher cache on the same batches.
    Returns (ce, kl) with kl=None when no sample in the split has a cache."""
    total_ce = 0.0
    total_kl = 0.0
    n = 0
    n_kl = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        b = make_batch_aligned(tokenizer, chunk, max_len, cache_index, top_k)
        if b is None:
            continue
        inputs, targets, mask, cache_batch, _ = b
        logits = model(inputs)
        ce = ce_loss_masked(logits, targets, mask)
        total_ce += ce.item()
        n += 1
        if cache_batch is not None:
            top_ids, top_logits, kl_mask = cache_batch
            kl = kl_loss_distill(logits, top_ids, top_logits, mask & kl_mask)
            total_kl += kl.item()
            n_kl += 1
    return total_ce / max(1, n), (total_kl / max(1, n_kl) if n_kl else None)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
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
    p.add_argument(
        "--distill-dir",
        default="data/teacher_logits/",
        help="Phase-1 teacher cache dir (manifest.json); unused when "
        "--distill-weight 0",
    )
    p.add_argument(
        "--distill-weight",
        type=float,
        default=0.5,
        help="fixed lambda mixing: loss = (1-l)*CE + l*KL; 0 = "
        "CE-only baseline (exact train_quantal_long behavior)",
    )
    p.add_argument("--top-k", type=int, default=64, help="must match the cache")
    args = p.parse_args()

    if "MLX_CUDA_GRAPH_CACHE_SIZE" not in os.environ:
        os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "2000"
        print("  note: MLX_CUDA_GRAPH_CACHE_SIZE unset — defaulted to 2000")
    else:
        print(
            f"  note: MLX_CUDA_GRAPH_CACHE_SIZE={os.environ['MLX_CUDA_GRAPH_CACHE_SIZE']} (respected)"
        )

    distill_active = args.distill_weight > 0
    if distill_active and not 0.0 < args.distill_weight <= 1.0:
        sys.exit(f"FATAL: --distill-weight {args.distill_weight} must be in (0, 1]")

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("  quantal distill training (deployed forward, masked CE + KL)")
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
    print(
        f"  distill:   {'ON' if distill_active else 'OFF'}  "
        f"lambda={args.distill_weight}  top_k={args.top_k}  "
        f"dir={args.distill_dir if distill_active else '-'}"
    )
    print(f"  metal:     {mx.metal.is_available()}")

    cache_index = None
    manifest = None
    if distill_active:
        print("0. loading teacher cache index...")
        cache_index, manifest = load_cache_index(args.distill_dir, args.top_k)
        print(
            f"   {len(cache_index)} cached samples (teacher={manifest.get('teacher')}, "
            f"tokenizer_id={manifest.get('tokenizer_id')})"
        )

    print("1. loading base model...")
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    from mlx.utils import tree_flatten

    tp = model.trainable_parameters()
    n_params = sum(v.nbytes for _, v in tree_flatten(tp)) / 1e6
    print(f"   loaded {n_params:.1f}M params")

    if manifest is not None:
        stud_tok = getattr(tokenizer, "vocab_size", None)
        if stud_tok is None:
            stud_tok = (
                len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else "unknown"
            )
        cache_tok = manifest.get("tokenizer_id")
        if cache_tok not in (None, "unknown") and str(cache_tok) != str(stud_tok):
            print(
                f"   WARNING: cache tokenizer_id={cache_tok} != student vocab_size={stud_tok} "
                "(byte-identical tokenizer expected — re-check Phase 1)"
            )

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
    last_kl = {
        "value": None
    }  # KL of the current step, read post-hoc (no extra forward)

    def loss_fn(params):
        model.update(params)
        logits = model(inputs)
        ce = ce_loss_masked(logits, targets, mask)
        if distill_active and cache_batch is not None:
            top_ids, top_logits, kl_mask = cache_batch
            kl = kl_loss_distill(logits, top_ids, top_logits, mask & kl_mask)
            last_kl["value"] = kl
            return (1.0 - args.distill_weight) * ce + args.distill_weight * kl
        last_kl["value"] = None
        return ce

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_kl = 0.0
        kl_steps = 0
        missing_total = 0
        steps_done = 0
        idxs = np.random.permutation(len(train_samples))

        for batch_idx in range(0, len(idxs), args.batch_size):
            batch_idxs = idxs[batch_idx : batch_idx + args.batch_size]
            b = make_batch_aligned(
                tokenizer,
                [train_samples[i] for i in batch_idxs],
                args.max_len,
                cache_index if distill_active else None,
                args.top_k,
            )
            if b is None:
                continue
            inputs, targets, mask, cache_batch, n_missing = b
            missing_total += n_missing

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
            if distill_active and last_kl["value"] is not None:
                epoch_kl += last_kl[
                    "value"
                ].item()  # KL of this step (no extra forward)
                kl_steps += 1

            if global_step % 50 == 0:
                print(
                    f"   step {global_step}/{total_steps} | loss {lv:.4f} | lr {float(schedule(global_step)):.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, steps_done)
        avg_kl = epoch_kl / max(1, kl_steps) if distill_active and kl_steps else None
        elapsed = time.time() - t0
        sps = steps_done / max(1.0, elapsed)

        # Eval with the same batch size as training: the CUDA graph cache is
        # warmed on training shapes, so a smaller eval batch would JIT-build new
        # graph shapes under memory pressure and cublasLtMatmul can fail (code 7).
        if distill_active:
            val_loss, val_kl = evaluate_distill(
                model,
                tokenizer,
                val_samples,
                args.max_len,
                args.batch_size,
                cache_index,
                args.top_k,
            )
        else:
            val_loss = evaluate(
                model, tokenizer, val_samples, args.max_len, batch_size=args.batch_size
            )
            val_kl = None
        lr_now = float(schedule(global_step))

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "kl_loss": round(avg_kl, 4) if avg_kl is not None else None,
            "val_kl": round(val_kl, 4) if val_kl is not None else None,
            "distill_active": distill_active,
            "distill_weight": args.distill_weight if distill_active else 0.0,
            "missing_cache": missing_total,
            "lr": lr_now,
            "elapsed_s": round(elapsed, 1),
            "steps_per_sec": round(sps, 2),
            "best_val": round(best_val, 4),
            "global_step": global_step,
        }
        with open(args.curve, "a") as f:
            f.write(json.dumps(rec) + "\n")

        kl_s = f" | kl {avg_kl:.4f}" if avg_kl is not None else ""
        kl_v = f" | val_kl {val_kl:.4f}" if val_kl is not None else ""
        miss_s = f" | missing_cache {missing_total}" if missing_total else ""
        print(
            f"  epoch {epoch + 1}/{args.epochs} | loss {avg_loss:.4f}{kl_s} | val {val_loss:.4f}{kl_v}{miss_s} | lr {lr_now:.2e} | {elapsed:.1f}s ({sps:.2f} steps/s)",
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
