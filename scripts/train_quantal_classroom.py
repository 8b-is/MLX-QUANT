#!/usr/bin/env python3
"""
train_quantal_classroom.py — ring-of-teachers (multi-faculty) logits-KL
distillation for the quantal brain. Extends train_quantal_distill.py: instead
of ONE teacher cache, a FACULTY of tokenizer-matched teachers contributes a
per-sample consensus KL term. Optional seminar corpus flows into the masked CE.

Loss:
    loss = α·masked_CE(corpus + seminar) + β·mean_f KL(student ∥ teacher_f)

α = --ce-weight, β = --faculty-weight (defaults 0.5/0.5). With
--beta-ramp-epochs N > 0, β ramps linearly from 0 to its configured value over
the first N epochs (CE-only warmup so the quantizer stabilizes first).

Consensus KL: per sample, each faculty's sparse top-k cache entry is loaded
for the sample hash. The student's KL is computed against EACH present faculty
on that faculty's own top-k ids (log-softmax form, exactly kl_loss_distill),
then AVERAGED over the present faculty — so the student binds to the ring's
mean teaching signal, not any single teacher's errors. Missing cache for a
faculty → that faculty contributes nothing for that sample (masked out, never
crash). All faculty missing → KL skipped (CE only).

Faculty caches: one directory per teacher, each with the sparse top-k npz
cache + manifest.json (top_k, tokenizer_id, teacher) produced by
teacher_logits.py. --faculty-dirs is a space-separated list.

Seminar: --seminar-data (optional) is a jsonl of teacher-generated text
(tokenizer-mismatched teachers are fine — we keep only the text) merged into
the masked-CE term via the same load_jsonl path.

Student forward is the deployed-forward thresholded ternary BitLinear —
unchanged. The parity invariant (training == Rust runner) is untouched: only
the loss changed.

NOTES for HF Jobs (no CUDA toolkit in the container):
  - MLX_CUDA_GRAPH_CACHE_SIZE is deliberately NOT defaulted here — the wrapper
    (h200_train.py) sets it when the synthetic CUDA_HOME is in place.
"""

import argparse
import hashlib
import json
import math  # noqa: F401  (kept for parity with the single-teacher script)
import os
import sys
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.nn.layers.bitlinear import BitLinear  # noqa: E402
from mlx.utils import tree_map
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
    """Tokenize + filter (<2 tokens dropped) + x64 bucket + pad + roll shim.
    Returns (inputs, targets, mask, kept_texts) where kept_texts are the rows
    that actually landed in the batch (cache rows must align 1:1 with them)."""
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
    # fork-mlx uses `axis=`, stock mlx uses `axes=` — shim both.
    try:
        targets = mx.roll(inputs, -1, axis=(1,))
    except TypeError:
        targets = mx.roll(inputs, -1, axes=(1,))
    targets[:, -1] = 0  # pad token 0, weighted out by mask
    return inputs, targets, mask, kept


def make_batch(tokenizer, texts: list[str], max_len: int):
    b = _build_batch(tokenizer, texts, max_len)
    if b is None:
        return None
    inputs, targets, mask, _ = b
    return inputs, targets, mask


def make_batch_faculty(
    tokenizer, texts: list[str], max_len: int, faculty_indices, top_k: int
):
    """make_batch + per-faculty cache rows aligned to the SAME batch rows.
    Returns (inputs, targets, mask, faculty_batch|None, n_missing) where
    faculty_batch is a list over faculty of (top_ids, top_logits, kl_mask)
    tuples (each shaped like the single-teacher cache_batch)."""
    b = _build_batch(tokenizer, texts, max_len)
    if b is None:
        return None
    inputs, targets, mask, kept = b
    faculty_batch = None
    n_missing = 0
    if faculty_indices is not None:
        faculty_batch = []
        for cache_index in faculty_indices:
            fb, nm = build_faculty_cache_batch(
                kept, inputs.shape[1], cache_index, top_k
            )
            n_missing += nm
            faculty_batch.append(fb)
    return inputs, targets, mask, faculty_batch, n_missing


def ce_loss_masked(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Masked cross-entropy: pad positions weighted out, honest mean."""
    vocab = logits.shape[-1]
    lg = logits.reshape(-1, vocab)
    tg = targets.reshape(-1)
    mk = mask.reshape(-1).astype(mx.float32)
    return nn.losses.cross_entropy(lg, tg, reduction="mean", weights=mk)


# ---------------------------------------------------------------------------
# 1b. Faculty-cache loading + consensus masked KL
# ---------------------------------------------------------------------------


def load_cache_index(distill_dir: str, top_k: int):
    """Build {sha1: npz path} from a faculty manifest; validate top_k.
    Returns (index, manifest)."""
    manifest_path = os.path.join(distill_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(
            f"FATAL: {manifest_path} not found — run scripts/teacher_logits.py "
            "first (classroom Phase 1)."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("top_k") != top_k:
        sys.exit(
            f"FATAL: cache top_k={manifest.get('top_k')} != --top-k {top_k} in {distill_dir}. "
            "The cache was generated with a different top-k — regenerate it or "
            f"pass --top-k {manifest.get('top_k')}."
        )
    index = {}
    for f in manifest["files"]:
        index[f["sha1"]] = os.path.join(distill_dir, f["file"])
    return index, manifest


def load_faculty(faculty_dirs: list[str], top_k: int, student_vocab: int):
    """Load all faculty indices + manifests; hard-error on tokenizer mismatch.
    Returns (faculty_indices, faculty_manifests)."""
    indices = []
    manifests = []
    for d in faculty_dirs:
        idx, man = load_cache_index(d, top_k)
        tid = man.get("tokenizer_id")
        if tid not in (None, "unknown") and str(tid) != str(student_vocab):
            sys.exit(
                f"FATAL: faculty {d} tokenizer_id={tid} != student vocab_size={student_vocab} "
                "(byte-identical tokenizer required for logits-KL)."
            )
        indices.append(idx)
        manifests.append(man)
        print(
            f"   faculty {man.get('teacher', d)}: {len(idx)} cached samples "
            f"(tokenizer_id={tid})"
        )
    return indices, manifests


def build_faculty_cache_batch(texts: list[str], max_t: int, cache_index, top_k: int):
    """Stack per-sample caches into (B, max_t, K) rows for ONE faculty member,
    aligned to the batch produced for the SAME texts. A sample without a cache
    file gets a zero row with mask all-False — its KL term is dropped (weighted
    0), never a crash. Returns ((top_ids, top_logits, kl_mask), n_missing)."""
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
        try:
            with np.load(path) as z:
                c_ids = np.asarray(z["top_ids"][:max_t], dtype=np.int32)
                c_lg = np.asarray(z["top_logits"][:max_t], dtype=np.float32)
                c_mk = np.asarray(z["mask"][:max_t], dtype=bool)
        except (EOFError, OSError, ValueError, KeyError):
            # corrupt/empty cache entry — treat as missing (KL dropped for
            # this sample), never crash the training loop.
            n_missing += 1
            rows.append(
                (
                    mx.zeros((max_t, top_k), dtype=mx.int32),
                    mx.zeros((max_t, top_k), dtype=mx.float32),
                    mx.zeros((max_t,), dtype=mx.bool_),
                )
            )
            continue
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


def kl_loss_faculty(
    student_logits: mx.array,
    faculty_batch: list,
    mask: mx.array,
) -> Optional[mx.array]:
    """Consensus KL: for each present faculty member, per-position KL on that
    faculty's own top-k ids (identical math to the single-teacher
    kl_loss_distill), then AVERAGE over the faculty members that have a
    non-empty kl_mask for the batch. Returns a scalar, or None when no faculty
    member has any valid position."""
    present = []
    for top_ids, top_logits, kl_mask in faculty_batch:
        if top_ids is None:
            continue
        if not bool(mx.any(kl_mask)):
            continue
        present.append((top_ids, top_logits, kl_mask))
    if not present:
        return None
    kl_sum = None
    kl_count = 0
    for top_ids, top_logits, kl_mask in present:
        ids = mx.clip(top_ids, 0, student_logits.shape[-1] - 1).astype(mx.int32)
        s_top = mx.take_along_axis(student_logits, ids, axis=-1)  # (B, T, K)
        t_lse = top_logits - mx.logsumexp(top_logits, axis=-1, keepdims=True)
        s_lse = s_top - mx.logsumexp(s_top, axis=-1, keepdims=True)
        p_t = mx.exp(t_lse)
        kl = mx.sum(p_t * (t_lse - s_lse), axis=-1)  # (B, T)
        mk = (mask & kl_mask).astype(mx.float32)
        denom = mx.maximum(mx.sum(mk), 1e-6)
        k = mx.sum(kl * mk) / denom
        kl_sum = k if kl_sum is None else kl_sum + k
        kl_count += 1
    if kl_count == 0:
        return None
    return kl_sum / kl_count


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
# 4. Batched masked eval (CE + faculty consensus KL)
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


def evaluate_faculty(
    model, tokenizer, samples, max_len, batch_size, faculty_indices, top_k
):
    """evaluate + consensus KL against the faculty caches on the same batches.
    Returns (ce, kl) with kl=None when no faculty member has a cache on the
    split."""
    total_ce = 0.0
    total_kl = 0.0
    n = 0
    n_kl = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        b = make_batch_faculty(tokenizer, chunk, max_len, faculty_indices, top_k)
        if b is None:
            continue
        inputs, targets, mask, faculty_batch, _ = b
        logits = model(inputs)
        ce = ce_loss_masked(logits, targets, mask)
        total_ce += ce.item()
        n += 1
        if faculty_batch is not None:
            kl = kl_loss_faculty(logits, faculty_batch, mask)
            if kl is not None:
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
    p.add_argument(
        "--seminar-data",
        default=None,
        help="optional jsonl of teacher-generated text (merged into masked CE)",
    )
    p.add_argument(
        "--faculty-dirs",
        nargs="+",
        required=True,
        help="space-separated list of faculty cache dirs (each has manifest.json)",
    )
    p.add_argument(
        "--ce-weight", type=float, default=0.5, help="α in loss = α·CE + β·KL"
    )
    p.add_argument(
        "--faculty-weight", type=float, default=0.5, help="β in loss = α·CE + β·KL"
    )
    p.add_argument(
        "--beta-ramp-epochs",
        type=int,
        default=0,
        help="N>0: ramp β linearly from 0 over the first N epochs (CE warmup)",
    )
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
        "per-projection-RMSNorm BitLinear forward.",
    )
    p.add_argument("--top-k", type=int, default=64, help="must match the caches")
    args = p.parse_args()

    # NOTE: MLX_CUDA_GRAPH_CACHE_SIZE deliberately NOT defaulted here — the
    # HF-Jobs wrapper sets it when the synthetic CUDA_HOME is in place.

    if not 0.0 < args.faculty_weight <= 1.0:
        sys.exit(f"FATAL: --faculty-weight {args.faculty_weight} must be in (0, 1]")
    if not 0.0 <= args.ce_weight <= 1.0:
        sys.exit(f"FATAL: --ce-weight {args.ce_weight} must be in [0, 1]")

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("  quantal classroom training (deployed forward, masked CE + faculty KL)")
    print(f"  model:     {args.model}")
    print(f"  data:      {args.data}")
    print(f"  seminar:   {args.seminar_data or '-'}")
    print(f"  faculty:   {args.faculty_dirs}")
    print(
        f"  weights:   CE {args.ce_weight} / faculty {args.faculty_weight}"
        f"{f' (ramp {args.beta_ramp_epochs} ep)' if args.beta_ramp_epochs else ''}"
    )
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

    stud_tok = getattr(tokenizer, "vocab_size", None)
    if stud_tok is None:
        stud_tok = len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else "unknown"

    print("0. loading faculty cache indices...")
    faculty_indices, faculty_manifests = load_faculty(
        args.faculty_dirs, args.top_k, stud_tok
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
    if args.seminar_data:
        sem = load_jsonl(args.seminar_data)
        if sem:
            samples = samples + sem
            print(f"   seminar merged: +{len(sem)} samples -> {len(samples)} total")
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
    last_kl = {"value": None}

    def loss_fn(params):
        model.update(params)
        logits = model(inputs)
        ce = ce_loss_masked(logits, targets, mask)
        # β ramp: linear 0 -> faculty_weight over the first beta_ramp_epochs
        if args.beta_ramp_epochs > 0:
            ep = min(epoch, args.beta_ramp_epochs)
            beta = args.faculty_weight * (ep / args.beta_ramp_epochs)
            alpha = args.ce_weight + (
                args.faculty_weight - beta
            ) * args.ce_weight / max(1e-9, args.faculty_weight)
            alpha = min(1.0, alpha) if args.ce_weight > 0 else 0.0
        else:
            beta = args.faculty_weight
            alpha = args.ce_weight
        if faculty_batch is not None:
            kl = kl_loss_faculty(logits, faculty_batch, mask)
            if kl is not None:
                last_kl["value"] = kl
                norm = alpha + beta
                return (alpha / norm) * ce + (beta / norm) * kl
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
            b = make_batch_faculty(
                tokenizer,
                [train_samples[i] for i in batch_idxs],
                args.max_len,
                faculty_indices,
                args.top_k,
            )
            if b is None:
                continue
            inputs, targets, mask, faculty_batch, n_missing = b
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
            if last_kl["value"] is not None:
                epoch_kl += last_kl["value"].item()
                kl_steps += 1

            if global_step % 50 == 0:
                print(
                    f"   step {global_step}/{total_steps} | loss {lv:.4f} | lr {float(schedule(global_step)):.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, steps_done)
        avg_kl = epoch_kl / max(1, kl_steps) if kl_steps else None
        elapsed = time.time() - t0
        sps = steps_done / max(1.0, elapsed)

        val_loss, val_kl = evaluate_faculty(
            model,
            tokenizer,
            val_samples,
            args.max_len,
            args.batch_size,
            faculty_indices,
            args.top_k,
        )
        lr_now = float(schedule(global_step))

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "faculty_kl": round(avg_kl, 4) if avg_kl is not None else None,
            "val_faculty_kl": round(val_kl, 4) if val_kl is not None else None,
            "faculty_weight": round(args.faculty_weight, 4),
            "ce_weight": round(args.ce_weight, 4),
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
