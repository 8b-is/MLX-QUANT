#!/usr/bin/env python3
"""tokenizer_identity_test.py — Phase A gate: does the Qwen3 tokenizer produce
byte-identical IDs to Qwen2.5 on the quantal corpus?

If yes, CE is comparable across bases and the Qwen3-1.7B/4B ladder is directly
predictive of the old 0.5597 / 2.1469 lines. If no, re-tokenization enters scope.

Usage:  python3 tokenizer_identity_test.py [--n 100] [--seed 42]
"""

import argparse
import json
import random

from transformers import AutoTokenizer


def load_samples(path: str, n: int, seed: int):
    rng = random.Random(seed)
    texts = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", obj.get("content", obj.get("instruction", "")))
            if text:
                texts.append(text)
    rng.shuffle(texts)
    return texts[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/train_ultra_qwen3.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = load_samples(args.corpus, args.n, args.seed)
    print(f"  samples: {len(samples)} from {args.corpus}")

    tok_old = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tok_new = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    print(f"  old vocab: {tok_old.vocab_size}  new vocab: {tok_new.vocab_size}")
    print(f"  old eos: {tok_old.eos_token_id}  new eos: {tok_new.eos_token_id}")

    identical = 0
    total_tokens = 0
    mismatches = []
    for i, s in enumerate(samples):
        ids_old = tok_old.encode(s)
        ids_new = tok_new.encode(s)
        if ids_old == ids_new:
            identical += 1
        else:
            mismatches.append((i, s[:80], ids_old[:16], ids_new[:16]))
        total_tokens += len(ids_old)

    print(f"\n  identical ID sequences: {identical}/{len(samples)}")
    print(f"  mean tokens/sample: {total_tokens / max(1, len(samples)):.1f}")
    if mismatches:
        print(f"\n  first {min(3, len(mismatches))} mismatches:")
        for i, s, a, b in mismatches[:3]:
            print(f"   sample {i}: {s!r}")
            print(f"     old: {a}")
            print(f"     new: {b}")
        print("\n  RESULT: FAIL — tokenizer differs, CE not directly comparable")
    else:
        print("\n  RESULT: PASS — byte-identical IDs, CE comparable across bases")


if __name__ == "__main__":
    main()
