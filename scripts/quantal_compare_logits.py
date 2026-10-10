#!/usr/bin/env python3
"""
quantal_compare_logits.py — GOLDEN-LOGITS GATE.

Compares the Rust runner's final-token logits
(`cargo run -p ternary --example quantal_logits -- <model_dir> <rust.json>`)
against the MLX reference (`quantal_golden_logits.py --out ref.json`).

Acceptance: agreement within ~1e-2 (reported as both max abs delta and max
relative delta; the verdict uses whichever tolerance is satisfied — oracle's
"~1e-2 relative or abs"). Also reports the vanilla-vs-bitlinear delta so the
oracle can see what the missing activation-RMSNorm/quant/biases cost.

Usage:
  python scripts/quantal_compare_logits.py rust.json ref.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def stats(name, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(
            "logits must be nonempty one-dimensional vectors of equal shape"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("logits must be finite")
    d = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1.0)
    rel = d / denom
    return {
        "name": name,
        "n": int(a.size),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float(rel.max()),
        "argmax_a": int(a.argmax()),
        "argmax_b": int(b.argmax()),
        "argmax_same": bool(a.argmax() == b.argmax()),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    rust_path, ref_path = sys.argv[1], sys.argv[2]
    rust = json.load(open(rust_path))
    ref = json.load(open(ref_path))

    # Which forward does the Rust runner implement? If the model dir carries
    # gains.f32/biases.f32 the runner is in faithful (per-projection RMSNorm +
    # activation_quant + biases) mode and must match the bitlinear reference;
    # otherwise it is weight-quant-only and must match the vanilla reference.
    faithful = (Path(rust["model_dir"]) / "gains.f32").exists()
    ref_key = "logits_bitlinear" if faithful else "logits_vanilla"
    mode = "faithful-bitlinear" if faithful else "weight-quant-only"
    print(f"  mode: {mode} (reference: {ref_key})")

    if not rust["prompts"] or len(rust["prompts"]) != len(ref["prompts"]):
        raise ValueError(
            "prompt sets must be nonempty and have equal length: "
            f"rust {len(rust['prompts'])} vs ref {len(ref['prompts'])}"
        )
    ok_all = True
    for rp, fp in zip(rust["prompts"], ref["prompts"]):
        if rp["prompt"] != fp["prompt"]:
            raise ValueError(
                f"prompt identity mismatch: rust {rp['prompt']!r} vs ref {fp['prompt']!r}"
            )
        if rp["n_tokens"] != fp["n_tokens"]:
            raise ValueError(
                f"prompt {rp['prompt']!r} token count mismatch: "
                f"rust {rp['n_tokens']} vs ref {fp['n_tokens']}"
            )
        rs = stats(
            f"prompt {rp['prompt']} rust-vs-mlx-{mode}", rp["logits"], fp[ref_key]
        )
        print(json.dumps(rs))
        gate = rs["max_abs"] <= 1e-2 or rs["max_rel"] <= 1e-2
        ok_all &= gate
        print(
            f"  GATE {'PASS' if gate else 'FAIL'} "
            f"(max_abs={rs['max_abs']:.3e} max_rel={rs['max_rel']:.3e})"
        )
        # Reference sanity: report the opposite pairing + vanilla-vs-bitlinear
        # gap so the oracle can see the delta the other forward would cost.
        other_key = "logits_vanilla" if faithful else "logits_bitlinear"
        if other_key in fp:
            os_ = stats(
                f"prompt {rp['prompt']} rust-vs-mlx-{other_key}",
                rp["logits"],
                fp[other_key],
            )
            print(json.dumps(os_))
        if "logits_bitlinear" in fp and "logits_vanilla" in fp:
            vs = stats(
                f"prompt {rp['prompt']} mlx-vanilla-vs-bitlinear",
                fp["logits_vanilla"],
                fp["logits_bitlinear"],
            )
            print(json.dumps(vs))

    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Invalid comparison input: {exc}", file=sys.stderr)
        sys.exit(1)
