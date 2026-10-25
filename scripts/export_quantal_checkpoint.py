#!/usr/bin/env python3
"""
export_quantal_checkpoint.py — re-export a trained quantal safetensors
checkpoint to the ayeOS {n+-1-<△>} ternary matrix format WITHOUT retraining.

train_quantal.py saves the full-precision BitLinear weights via
``model.save_weights()``; this tool reconstructs the same BitLinear
architecture (base HF model + Linear->BitLinear swap), loads the checkpoint,
and runs the identical ternary quantization as ``export_to_ayeos`` —
``weight_quant`` + ``mx.quantize(..., bits=2, mode='ternary')`` from the
MLX-QUANT fork (mlx with native ternary support). It can either write the
single ayeOS capsule JSON or split straight into the per-matrix files +
index.json layout the pocoo quantal demo viewer expects.

Requires the fork MLX build (mx.quantize with mode="ternary") — stock mlx
raises "Invalid quantization mode 'ternary'".

Usage:
  python scripts/export_quantal_checkpoint.py \
    --checkpoint quantal_model.safetensors \
    --out-dir demos/quantal \
    --model Qwen/Qwen2.5-0.5B \
    --group-size 64
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlx.nn.layers.bitlinear import BitLinear, weight_quant  # noqa: E402
from train_quantal import replace_linear_with_bitlinear  # noqa: E402


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ternary_quantize_numpy(w: np.ndarray, group_size: int, threshold: float = 0.5):
    """ayeOS ternary quantization, fork-`mx.quantize(mode='ternary')`-equivalent.

    ``w`` is the *full-precision* weight (NOT pre-quantized). Per
    ``group_size``-column group: scale = mean(|w|) — the same scale the
    training forward uses — and code = the thresholded ternary of ``w/scale``
    (zero when |w| < threshold·scale), packed 2-bit LSB-first, 16 codes per
    ``u32`` word. Mirrors what the Rust runner dequantizes as ``(code-1)*scale``
    and reproduces the forward's quantized weight exactly.

    Returns ``(codes, scales)`` numpy arrays matching the ayeOS schema.
    """
    n, k = w.shape
    codes = np.zeros((n, k // 16), dtype=np.uint32)
    scales = np.zeros((n, k // group_size), dtype=np.float32)
    for r in range(n):
        row = w[r]
        for g in range(k // group_size):
            group = row[g * group_size : (g + 1) * group_size]
            scale = np.abs(group).mean()
            scales[r, g] = scale
            vals = np.where(np.abs(group) < threshold * scale, 0, np.sign(group))
            vals = vals.astype(np.int8)
            for j, v in enumerate(vals):
                code = v + 1  # -1→0, 0→1, +1→2
                words = (g * group_size + j) // 16
                bit = 2 * ((g * group_size + j) % 16)
                codes[r, words] |= np.uint32(code) << bit
    return codes, scales


def export_capsule(model, group_size, metadata):
    """Same per-layer logic as train_quantal.export_to_ayeos, returned as a
    list of matrix entries instead of written to one giant JSON file."""
    entries = []
    for path, m in model.named_modules():
        if isinstance(m, BitLinear) and path:
            w = m["weight"]
            w_np = np.array(w.astype(mx.float32).tolist(), dtype=np.float32)
            codes, scales = ternary_quantize_numpy(w_np, group_size)
            entries.append(
                {
                    "name": path,
                    "dim": w.shape[0],
                    "in_features": w.shape[1],
                    "group_size": group_size,
                    "codes": codes.reshape(-1).tolist(),
                    "scales": scales.reshape(-1).tolist(),
                    "seed_hash": "quantal-trained",
                }
            )
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="trained quantal safetensors")
    p.add_argument(
        "--out-dir", required=True, help="pocoo demo dir for m*.json + index.json"
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="base HF model for architecture reconstruction",
    )
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--final-loss", type=float, default=None)
    p.add_argument("--val-loss", type=float, default=None)
    p.add_argument(
        "--base-model",
        default=None,
        help="base model recorded in the checkpoint capsule (may differ from --model)",
    )
    p.add_argument(
        "--source",
        default="vast.ai remote GPU run",
        help="where the checkpoint came from",
    )
    args = p.parse_args()

    print(f"  mlx: {mx.__version__}")
    print(f"  metal: {mx.metal.is_available()}")

    print("1. loading base model...")
    from mlx_lm.utils import load as mlx_load

    model, _ = mlx_load(args.model)
    print(f"   loaded {args.model}")

    print("2. swapping Linear -> BitLinear...")
    n_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    model = replace_linear_with_bitlinear(model)
    n_after = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(f"   replaced {n_before} Linear -> {n_after} BitLinear")

    print("3. loading checkpoint weights...")
    model.load_weights(args.checkpoint)
    print(f"   weights <- {args.checkpoint}")

    print("4. exporting to ayeOS ternary format...")
    entries = export_capsule(model, args.group_size, None)
    print(f"   exported {len(entries)} matrices")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("5. hashing checkpoint...")
    ckpt_sha = sha256_of(args.checkpoint)
    ckpt_size = Path(args.checkpoint).stat().st_size
    print(f"   sha256 {ckpt_sha}")
    print(f"   size   {ckpt_size}")

    print("6. writing per-matrix files + index.json...")
    now = int(time.time())
    matrices = []
    for i, e in enumerate(entries):
        fname = f"m{i:03d}.json"
        with open(out / fname, "w") as f:
            json.dump(e, f, separators=(",", ":"))
        matrices.append(
            {
                "file": fname,
                "name": e["name"],
                "dim": e["dim"],
                "in_features": e["in_features"],
                "group_size": e["group_size"],
                "bytes": (out / fname).stat().st_size,
            }
        )

    index = {
        "capsule_id": f"quantal-{now}",
        "address": {
            "role": "matrix.ternary.quantal",
            "tags": ["quantal", "bitnet", "ternary", "ayeos"],
        },
        "payload_type": "ternary_matrix",
        "metadata": {
            "base_model": args.base_model or args.model,
            "reconstruction_model": args.model,
            "binary": False,
            "group_size": args.group_size,
            "epochs": args.epochs,
            "final_loss": args.final_loss,
            "val_loss": args.val_loss,
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_size_bytes": ckpt_size,
            "checkpoint_source": args.source,
            "export_tool": "MLX-QUANT fork mlx "
            + mx.__version__
            + " (ternary quantize)",
            "export_complete": True,
        },
        "timestamp": now,
        "matrices": matrices,
    }
    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"   wrote {len(matrices)} matrix files + index.json -> {out}")
    print("  done")


if __name__ == "__main__":
    main()
