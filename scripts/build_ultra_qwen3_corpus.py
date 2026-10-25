#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "polars", "numpy"]
# ///
"""
build_ultra_qwen3_corpus.py — merge the 26 HF-cache datasets + train_ultra3.jsonl
into a single SOTA corpus for the Qwen3-1.7B quantal continued-train.

Extraction is column-agnostic: it walks each snapshot's data files
(parquet / jsonl / json / csv), pulls the first usable text field
(`text` > `messages`-style lists > `prompt`/`instruction`/`injection` >
composed instruction+response / JBB Goal+Behavior), flattens chat lists to
`Role: content` lines, caps per dataset (guardrail / prompt-injection /
dialogue sets get the most budget), dedupes across everything, shuffles
deterministically and writes `train_ultra_qwen3.jsonl` (`{"text": ...}` rows —
the exact schema train_quantal_long.py's `load_jsonl` consumes).

Run via uv (pyarrow/polars come from the inline deps, nothing touches the
fork-mlx system python3):

  uv run --script scripts/build_ultra_qwen3_corpus.py

Output:
  data/train_ultra_qwen3.jsonl        merged corpus (~20-30K samples)
  data/train_ultra_qwen3_corpus.json  per-dataset stats + totals
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path

CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# dataset id (as in the cache dir name) -> per-dataset cap.
# Budget favors the guardrail / prompt-injection / dialogue sets (next round's
# guardrail layer) and dials down the giant SFT pools to keep the total in the
# ~20-30K band alongside the 14,330 train_ultra3 samples.
DATASET_CAPS = {
    "3nesdeniz/agentic-prompt-injection-5k": 800,
    "3nesdeniz/agentic-prompt-injection-boundary-pairs": 600,
    "3nesdeniz/english-daily-dialogues-10k": 800,
    "3nesdeniz/english-prompt-injection-3k": 800,
    "3nesdeniz/guardrail-hard-negatives": 800,
    "3nesdeniz/turkish-conversation-prompt-injection": 600,
    "3nesdeniz/turkish-daily-dialogues-5k": 600,
    "3nesdeniz/turkish-prompt-injection-1k": 600,
    "allenai/tulu-3-sft-mixture": 400,
    "allenai/WildChat-1M": 400,
    "CaiZhiTech/Evaluation-Dataset-of-AI-Agent-Security-Guardrails": 600,
    "deepset/prompt-injections": 600,
    "GuardrailsAI/detect-jailbreak": 600,
    "HuggingFaceH4/ultrachat_200k": 400,
    "HuggingFaceTB/cosmopedia": 400,
    "JailbreakBench/JBB-Behaviors": 400,
    "JailbreakV-28K/JailBreakV-28k": 400,
    "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1": 600,
    "nvidia/Nemotron-SFT-Agentic-v2": 400,
    "Open-Orca/SlimOrca": 400,
    "rubend18/ChatGPT-Jailbreak-Prompts": 600,
    "teknium/OpenHermes-2.5": 400,
    "xTRam1/safe-guard-prompt-injection": 600,
}

# Explicitly excluded (present in the cache but not training text):
#   - donghyunli/Meta-Llama-3-8B-KronQ-HG       (quantization benchmark .pt tensors)
#   - yangpei-comp/macosworld_intel             (VM disk images + scoreboard)
#   - studioburnside/mlx-local-inference-benchmarks (benchmark result reports)
SKIPPED = [
    "donghyunli/Meta-Llama-3-8B-KronQ-HG",
    "yangpei-comp/macosworld_intel",
    "studioburnside/mlx-local-inference-benchmarks",
]

ROLE_MAP = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "assistant": "Assistant",
    "system": "System",
    "tool": "Tool",
    "function": "Function",
}


def snapshot_dir(dataset_id: str) -> Path:
    dirname = "datasets--" + dataset_id.replace("/", "--")
    snaps = [
        p
        for p in glob.glob(os.path.join(CACHE, dirname, "snapshots", "*"))
        if not os.path.basename(p).startswith("refs")
    ]
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {dataset_id} in {CACHE}")
    snaps = [p for p in snaps if not os.path.basename(p) == "main"]
    return Path(
        snaps[0]
        if snaps
        else sorted(glob.glob(os.path.join(CACHE, dirname, "snapshots", "*")))[0]
    )


def flatten_messages(msgs) -> str:
    """Flatten a chat list (`role`/`content` or sharegpt `from`/`value`, or
    role-less alternating) into `Role: content` lines."""
    lines = []
    has_roles = any(isinstance(m, dict) and ("role" in m or "from" in m) for m in msgs)
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if has_roles:
            role = m.get("role") or m.get("from") or "user"
            label = ROLE_MAP.get(str(role).lower(), str(role))
        else:
            label = "User" if i % 2 == 0 else "Assistant"
        content = m.get("content") if has_roles else m.get("content")
        if content is None and "value" in m:
            content = m["value"]
        if content is None:
            continue
        text = str(content).strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def extract_text(obj) -> str | None:
    """Column-agnostic text extraction for one record dict."""
    if not isinstance(obj, dict):
        return None
    # 1. author-provided canonical `text`
    t = obj.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    # 2. chat-list fields (messages / conversation / conversations / dialogue /
    #    dialog / chat) — but NOT `turns` when a canonical text exists (handled
    #    above); `turns` is still a chat list and fine when it is all we have.
    for key in (
        "messages",
        "conversation",
        "conversations",
        "dialogue",
        "dialog",
        "chat",
        "turns",
    ):
        v = obj.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            flat = flatten_messages(v)
            if flat:
                return flat
    # 3. direct prompt-ish strings
    for key in ("prompt", "instruction", "jailbreak_query", "content"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 4. Nemotron-RL `injection` dict (goal + target_tool + context)
    inj = obj.get("injection")
    if isinstance(inj, dict):
        parts = [
            str(inj.get(k, "")).strip()
            for k in (
                "goal",
                "target_tool",
                "attack_category",
                "injection_vector",
                "context",
            )
        ]
        txt = "\n".join(p for p in parts if p)
        if txt:
            return txt
    # 5. composed instruction + response
    inst = obj.get("instruction") or obj.get("prompt")
    resp = obj.get("response") or obj.get("completion") or obj.get("output")
    if (
        isinstance(inst, str)
        and inst.strip()
        and isinstance(resp, str)
        and resp.strip()
    ):
        return f"Instruction: {inst.strip()}\nResponse: {resp.strip()}"
    # 6. JBB-Behaviors csv: Goal + Behavior
    goal, beh = obj.get("Goal"), obj.get("Behavior")
    if isinstance(goal, str) and goal.strip() and isinstance(beh, str) and beh.strip():
        return f"Goal: {goal.strip()}\nBehavior: {beh.strip()}"
    # 7. rubend18 jailbreak csv: Prompt
    pr = obj.get("Prompt")
    if isinstance(pr, str) and pr.strip():
        return pr.strip()
    return None


def iter_rows(dataset_id: str):
    """Yield record dicts from every data file of the dataset's snapshot."""
    snap = snapshot_dir(dataset_id)
    # parquet files (with streaming — big pools like WildChat/OpenHermes are GBs)
    for f in sorted(glob.glob(os.path.join(snap, "**", "*.parquet"), recursive=True)):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pyarrow missing — run via `uv run --script`")
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096):
            for rec in batch.to_pylist():
                yield rec
    for f in sorted(glob.glob(os.path.join(snap, "**", "*.jsonl"), recursive=True)):
        with open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    for f in sorted(glob.glob(os.path.join(snap, "**", "*.json"), recursive=True)):
        base = os.path.basename(f)
        if base in ("README.md", "dataset.json", "MANIFEST.json", "SHA256SUMS"):
            continue
        try:
            with open(f, "rt", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, list):
            for rec in obj:
                yield rec
        elif isinstance(obj, dict) and any(isinstance(v, list) for v in obj.values()):
            # some repos nest the records under a key
            for v in obj.values():
                if isinstance(v, list):
                    for rec in v:
                        yield rec
    for f in sorted(glob.glob(os.path.join(snap, "**", "*.csv"), recursive=True)):
        try:
            import polars as pl
        except ImportError:
            raise SystemExit("polars missing — run via `uv run --script`")
        try:
            df = pl.read_csv(f, infer_schema_length=200)
        except Exception:
            continue
        for rec in df.iter_rows(named=True):
            yield rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="data/train_ultra_qwen3.jsonl",
        help="output jsonl path (default: data/train_ultra_qwen3.jsonl)",
    )
    ap.add_argument(
        "--ultra3",
        default="data/train_ultra3.jsonl",
        help="previous corpus to merge (default: data/train_ultra3.jsonl)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="data/train_ultra_qwen3_corpus.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    corpus: dict[str, str] = {}  # sha1(text) -> text (dedupe across everything)

    def add(text: str) -> bool:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in corpus:
            return False
        corpus[h] = text
        return True

    # -- base corpus
    base_n = 0
    base_path = Path(args.ultra3)
    if base_path.exists():
        with open(base_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = extract_text(obj)
                if txt and add(txt):
                    base_n += 1
    print(f"base corpus {args.ultra3}: {base_n} samples")

    # -- datasets
    stats = {}
    for ds, cap in DATASET_CAPS.items():
        try:
            n_taken = 0
            n_seen = 0
            for rec in iter_rows(ds):
                n_seen += 1
                if n_taken >= cap:
                    continue
                txt = extract_text(rec)
                if txt and add(txt):
                    n_taken += 1
            stats[ds] = {"seen": n_seen, "taken": n_taken, "cap": cap}
            print(f"  {ds}: seen {n_seen}, taken {n_taken} (cap {cap})")
        except FileNotFoundError as e:
            stats[ds] = {"error": str(e)}
            print(f"  {ds}: SKIP ({e})")
        except Exception as e:  # keep the build alive on one bad dataset
            stats[ds] = {"error": str(e)}
            print(f"  {ds}: ERROR {e}")

    # -- shuffle + write
    texts = list(corpus.values())
    rng.shuffle(texts)
    with open(out, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    total_ds = sum(s.get("taken", 0) for s in stats.values())
    report = {
        "base_corpus": {"file": args.ultra3, "samples": base_n},
        "datasets": stats,
        "datasets_total_taken": total_ds,
        "datasets_skipped": SKIPPED,
        "dedup": {
            "unique_after_merge": len(texts),
            "removed_duplicates": base_n + total_ds - len(texts),
        },
        "final_total": len(texts),
        "out": str(out),
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  datasets total: {total_ds}")
    print(f"  unique after merge+dedupe: {len(texts)}")
    print(f"  wrote -> {out}")
    print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()
