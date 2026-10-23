#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "pyarrow"]
# ///
"""
build_corpus_v2.py — STREAMING corpus builder (v2) for the Qwen3-1.7B quantal
continued-train, designed to run ON THE GPU BOX (vast H100, Ubuntu, pip mlx
0.30.0, ~14GB free disk).

The v1 corpus (data/train_ultra_qwen3.jsonl, 20,007 samples) overfits after
epoch 2 (val 3.0563 -> 6.39) — a classic small-corpus signature. v2 triples the
per-dataset caps of the same 23 datasets (guardrail / prompt-injection /
dialogue emphasis kept), adds three new capped sets (Anthropic/hh-rlhf,
databricks/databricks-dolly-15k, HuggingFaceH4/no_robots), and merges the
existing data/train_ultra3.jsonl uncapped. Target: ~55-60k samples.

DISK-BOUNDED GUARANTEE (the box has only 14GB free):
  - datasets are streamed (`load_dataset(..., streaming=True)`), so only the
    records we actually keep are materialized — no full snapshot is written.
  - the hub/datasets cache is redirected to /tmp (ephemeral): we set
    HF_HUB_DISABLE_XET=1 (blocks the Xet storage backend from spooling big
    chunk caches) and pass --cache-dir /tmp/hf-datasets-cache to load_dataset,
    so nothing lands in the ~60GB root HF cache or a full snapshot.
  - the only persistent output is the corpus jsonl (~100-200MB for 40-60k
    samples) + the report json; the dedup set (sha1 -> text) lives in RAM.
  - once a dataset's cap is reached we BREAK the stream (v1 kept iterating and
    only skipped taking; on a network stream that wastes bandwidth).

Extraction is the v1 logic (column-agnostic: `text` > `messages`-style chat
lists > `prompt`/`instruction`/`injection` > composed instruction+response;
chat lists flatten to `Role: content` lines), plus a v2 addition for
Anthropic/hh-rlhf's whole-dialogue `chosen`/`rejected` strings. Dedup by
sha1(text) across everything; deterministic shuffle with
random.Random(seed).shuffle (same seed style as v1).

Output schema: `{"text": ...}` rows — the exact schema train_quantal_long.py /
train_quantal_distill.py's load_jsonl consumes.

Run ON THE BOX (downloads must land on the box, never a personal machine):

  pip3 install --break-system-packages datasets
  python3 scripts/build_corpus_v2.py \
    --out data/train_ultra_qwen3_v2.jsonl \
    --report data/train_ultra_qwen3_v2_corpus.json \
    --ultra3 data/train_ultra3.jsonl
  # smoke run (tiny caps, no network blast):
  python3 scripts/build_corpus_v2.py --max-per-dataset 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

# dataset id -> per-dataset cap.
# v2: the SAME 23 datasets as v1 with each cap roughly TRIPLED (the 23-dataset
# portion now sums to ~38.4k instead of 12.8k) — the guardrail / prompt-
# injection / dialogue sets keep the biggest share of the budget. The three
# new datasets are marked `# v2 new`.
DATASET_CAPS = {
    # --- guardrail / prompt-injection / dialogue (biggest share) ---
    "3nesdeniz/agentic-prompt-injection-5k": 2400,
    "3nesdeniz/agentic-prompt-injection-boundary-pairs": 1800,
    "3nesdeniz/english-daily-dialogues-10k": 2400,
    "3nesdeniz/english-prompt-injection-3k": 2400,
    "3nesdeniz/guardrail-hard-negatives": 2400,
    "3nesdeniz/turkish-conversation-prompt-injection": 1800,
    "3nesdeniz/turkish-daily-dialogues-5k": 1800,
    "3nesdeniz/turkish-prompt-injection-1k": 1800,
    "CaiZhiTech/Evaluation-Dataset-of-AI-Agent-Security-Guardrails": 1800,
    "deepset/prompt-injections": 1800,
    "GuardrailsAI/detect-jailbreak": 1800,
    "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1": 1800,
    "rubend18/ChatGPT-Jailbreak-Prompts": 1800,
    "xTRam1/safe-guard-prompt-injection": 1800,
    "JailbreakBench/JBB-Behaviors": 1200,
    "JailbreakV-28K/JailBreakV-28k": 1200,
    # --- large SFT / dialogue pools (dialed down) ---
    "allenai/tulu-3-sft-mixture": 1200,
    "allenai/WildChat-1M": 1200,
    "HuggingFaceH4/ultrachat_200k": 1200,
    "HuggingFaceTB/cosmopedia": 1200,
    "nvidia/Nemotron-SFT-Agentic-v2": 1200,
    "Open-Orca/SlimOrca": 1200,
    "teknium/OpenHermes-2.5": 1200,
    # --- v2 new (security/agentic/instruction-relevant) ---
    "Anthropic/hh-rlhf": 1500,  # v2 new (helpful+harmless)
    "databricks/databricks-dolly-15k": 1000,  # v2 new
    "HuggingFaceH4/no_robots": 1500,  # v2 new
}

# datasets whose default config is not the only useful one: stream each config
# under the SAME dataset cap.
MULTI_CONFIG = {
    "Anthropic/hh-rlhf": ["helpful", "harmless"],
}

# Explicitly excluded (present in the v1 cache but not training text) — kept
# for report continuity; v2 streams from the hub so these are moot but harmless.
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
        content = m.get("content")
        if content is None and "value" in m:
            content = m["value"]
        if content is None:
            continue
        text = str(content).strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def extract_text(obj) -> str | None:
    """Column-agnostic text extraction for one record dict (v1 logic, plus the
    v2 `chosen`/`rejected` addition for Anthropic/hh-rlhf)."""
    if not isinstance(obj, dict):
        return None
    # 1. author-provided canonical `text`
    t = obj.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    # 2. chat-list fields (messages / conversation / conversations / dialogue /
    #    dialog / chat / turns)
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
    # 3. composed instruction + response — checked BEFORE the single prompt-ish
    #    fields (v2 change: in v1 this branch was dead code, shadowed by the
    #    `prompt`/`instruction` tier; databricks-dolly-15k needs the full
    #    instruction+response pair, and no v1 dataset record has both a usable
    #    single prompt field AND a response without hitting the chat tiers)
    inst = obj.get("instruction") or obj.get("prompt")
    resp = obj.get("response") or obj.get("completion") or obj.get("output")
    if (
        isinstance(inst, str)
        and inst.strip()
        and isinstance(resp, str)
        and resp.strip()
    ):
        return f"Instruction: {inst.strip()}\nResponse: {resp.strip()}"
    # 4. direct prompt-ish strings
    for key in ("prompt", "instruction", "jailbreak_query", "content"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 5. Nemotron-RL `injection` dict (goal + target_tool + context)
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
    # 5b. v2 addition — hh-rlhf whole-dialogue strings ("Human: ...\n\nAssistant: ...")
    for key in ("chosen", "rejected"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 6. JBB-Behaviors csv: Goal + Behavior
    goal, beh = obj.get("Goal"), obj.get("Behavior")
    if isinstance(goal, str) and goal.strip() and isinstance(beh, str) and beh.strip():
        return f"Goal: {goal.strip()}\nBehavior: {beh.strip()}"
    # 7. rubend18 jailbreak csv: Prompt
    pr = obj.get("Prompt")
    if isinstance(pr, str) and pr.strip():
        return pr.strip()
    return None


def stream_records(dataset_id: str, cache_dir: str, config=None):
    """Yield record dicts from a dataset, STREAMING (datasets library).

    Downloads go to `cache_dir` (ephemeral /tmp by default), never to a full
    snapshot in ~/.cache/huggingface. `trust_remote_code=True` because several
    of these datasets (nvidia/Nemotron-*, ...) ship custom loader scripts.
    """
    from datasets import load_dataset

    kwargs = dict(streaming=True, cache_dir=cache_dir, trust_remote_code=True)
    if config is not None:
        kwargs["name"] = config
    try:
        ds = load_dataset(dataset_id, split="train", **kwargs)
    except Exception:
        # no `train` split (or other load quirk): fall back to the first split
        ds = load_dataset(dataset_id, **kwargs)
        if isinstance(ds, dict):  # IterableDatasetDict -> first split
            ds = ds[list(ds.keys())[0]]
    for rec in ds:
        yield rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="data/train_ultra_qwen3_v2.jsonl",
        help="output jsonl path (default: data/train_ultra_qwen3_v2.jsonl)",
    )
    ap.add_argument(
        "--report",
        default="data/train_ultra_qwen3_v2_corpus.json",
        help="per-dataset stats + totals (default: data/train_ultra_qwen3_v2_corpus.json)",
    )
    ap.add_argument(
        "--ultra3",
        default="data/train_ultra3.jsonl",
        help="previous corpus to merge uncapped (default: data/train_ultra3.jsonl)",
    )
    ap.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="safety override: cap every dataset at min(its cap, N)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stream-chunk",
        type=int,
        default=1000,
        help="samples per write batch flushed to the output jsonl",
    )
    ap.add_argument(
        "--cache-dir",
        default="/tmp/hf-datasets-cache",
        help="ephemeral datasets/hub cache dir (default: /tmp/hf-datasets-cache) — "
        "keeps the box's root HF cache untouched",
    )
    args = ap.parse_args()

    # disk-bounded: block the Xet storage backend (it spools large chunk caches)
    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("  note: HF_HUB_DISABLE_XET=1 (xet chunk caches off)")

    try:
        import datasets  # noqa: F401  (import must be deferred so the module
    except ImportError:  #                  imports on machines without datasets)
        raise SystemExit(
            "FATAL: the `datasets` library is required (streaming load). "
            "On the box:  pip3 install --break-system-packages datasets"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    corpus: dict[str, str] = {}  # sha1(text) -> text (dedupe across everything)
    raw_extracted = 0  # successful extractions (before dedup)

    def add(text: str) -> bool:
        nonlocal raw_extracted
        raw_extracted += 1
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in corpus:
            return False
        corpus[h] = text
        return True

    # -- base corpus (train_ultra3.jsonl, uncapped — already on the box)
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
        print(f"base corpus {args.ultra3}: {base_n} samples (uncapped)")
    else:
        print(f"base corpus {args.ultra3}: ABSENT — skipping (expected on the box)")

    # -- datasets (streamed, capped, break on cap)
    stats = {}
    for ds_id, cap in DATASET_CAPS.items():
        if args.max_per_dataset is not None:
            cap = min(cap, args.max_per_dataset)
        configs = MULTI_CONFIG.get(ds_id, [None])
        n_taken = 0
        n_seen = 0
        config_errors = []
        for cfg in configs:
            try:
                for rec in stream_records(ds_id, args.cache_dir, config=cfg):
                    n_seen += 1
                    if n_taken >= cap:
                        break  # v2: stop the stream once capped (bandwidth)
                    txt = extract_text(rec)
                    if txt and add(txt):
                        n_taken += 1
            except Exception as e:  # keep the build alive on one bad dataset
                config_errors.append(f"{cfg}: {e}")
                print(f"  {ds_id} [{cfg}]: ERROR {e}")
            if n_taken >= cap:
                break
        stats[ds_id] = {"seen": n_seen, "taken": n_taken, "cap": cap}
        if config_errors:
            stats[ds_id]["config_errors"] = config_errors
        print(f"  {ds_id}: seen {n_seen}, taken {n_taken} (cap {cap})", flush=True)

    # -- deterministic shuffle + chunked write
    texts = list(corpus.values())
    rng.shuffle(texts)
    with open(out, "w", encoding="utf-8") as fh:
        buf = []
        for t in texts:
            buf.append(json.dumps({"text": t}, ensure_ascii=False))
            if len(buf) >= args.stream_chunk:
                fh.write("\n".join(buf) + "\n")
                buf = []
        if buf:
            fh.write("\n".join(buf) + "\n")

    total_ds = sum(s.get("taken", 0) for s in stats.values())
    report = {
        "base_corpus": {"file": args.ultra3, "samples": base_n},
        "datasets": stats,
        "datasets_total_taken": total_ds,
        "datasets_skipped": SKIPPED,
        "dedup": {
            "raw_extracted": raw_extracted,
            "unique_after_merge": len(texts),
            "removed_duplicates": raw_extracted - len(texts),
        },
        "final_total": len(texts),
        "out": str(out),
        "cache_dir": args.cache_dir,
        "seed": args.seed,
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  datasets total: {total_ds}")
    print(f"  unique after merge+dedupe: {len(texts)}")
    print(f"  wrote -> {out}")
    print(f"  report -> {args.report}")
    print(f"  CORPUS_V2_DONE n={len(texts)} distinct={len(texts)}")


if __name__ == "__main__":
    main()
