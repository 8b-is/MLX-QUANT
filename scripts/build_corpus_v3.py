#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "pyarrow"]
# ///
"""
build_corpus_v3.py — STREAMING corpus builder (v3) for the quantal continued-train.

v2 (build_corpus_v2.py) produced a flat `{"text": ...}` corpus from 23 SFT /
guardrail datasets. v3 refocuses on AGENTIC + SAFETY + CYBER content and emits
the structured schema the next training stage consumes:

    {"id": str, "user_message": str, "free_response": str}

Datasets (14, each with a soft cap — missing/gated/failed sets are logged and
skipped, never fatal):

  agentic / RL   nvidia/Nemotron-Agentic-v1 (40k)
                 openbmb/UltraInteract_sft (25k)
                 nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 (15k)
                 nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 (8k)
  safety/guard   Lakera/mosscap_prompt_injection (30k)
                 yueliu1999/GuardReasonerTrain (20k)
                 nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0 (8k)
                 WNT3D/Ultimate-Offensive-Red-Team (8k)
  instruction    HuggingFaceTB/smoltalk (15k)
                 WizardLMTeam/WizardLM_evol_instruct_V2_196k (10k)
                 OpenCoder-LLM/opc-sft-stage2 (12k)
                 theblackcat102/evol-codealpaca-v1 (8k)
  cyber          ethz-spylab/ctf-satml24 (8k)
                 Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset (5k)

Conventions carried over from v2:
  - `load_dataset(..., streaming=True)` + ephemeral cache dir + HF_HUB_DISABLE_XET=1
    so nothing big lands on disk; only the records we keep are materialized.
  - break the stream once a dataset's cap is reached (no wasted bandwidth).
  - deterministic seed (42): records are taken in stream order up to the cap,
    then each dataset file is shuffled with random.Random(seed).
  - global dedup (across every dataset): sha1(user_message + "\\x00" + free_response),
    plus an id-set for datasets that carry real ids.

Normalization (column-agnostic, datasets differ wildly):
  1. per-dataset explicit field hints when a schema is known (dotted paths into
     nested dicts, e.g. `responses_create_params.input`).
  2. chat-list fields (`messages`, `conversation(s)`, `dialogue`, `history`, ...)
     split into user/assistant parts by role.
  3. instruction/prompt/query/question/user -> user_message;
     output/answer/response/completion/assistant -> free_response.
  4. generic fallback: first string column -> user_message, remaining -> free_response.
  id comes from `id`/`_id`/`idx`/`uuid`/`trajectory_id`/... else hash(user+free).

Upload: one jsonl per dataset staged in `--staging`, then `hf buckets cp` per file
into `hf://buckets/PeetPedro/quantal-corpus-v3/data/`. The `hf` CLI is used as a
subprocess so it uses ITS OWN authenticated token (the shell HF_TOKEN may be stale
and must not be relied on). NOTE: on hf CLI >= 1.22 `hf buckets cp` takes no
`--type` flag (bucket URIs are already typed by the `hf://buckets/...` scheme) —
do not pass `--type bucket`.

Run (from MLX-QUANT, per workspace convention everything python goes through uv):

  uv run --with datasets --with pyarrow python scripts/build_corpus_v3.py
  # smoke run (tiny caps, no upload):
  uv run --with datasets --with pyarrow python scripts/build_corpus_v3.py \
      --max-per-dataset 15 --no-upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# dataset specs: cap (soft), and optional explicit streams / field hints.
# `streams` = list of (config, split); absent -> default config, split "train"
# (falling back to the first available split). The cap is shared across a
# dataset's streams. `user`/`free` are dotted paths into nested dicts;
# `free` may be "@aiq_trace" (extract output strings from the AIQ trace list).
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    # --- agentic / RL ---
    "nvidia/Nemotron-Agentic-v1": {
        "cap": 40000,
        # interactive_agent / tool_calling are SPLITS of the default config,
        # not config names (tool_calling may fail on a parquet cast bug — fine,
        # it is logged and skipped).
        "streams": [(None, "interactive_agent"), (None, "tool_calling")],
    },
    "openbmb/UltraInteract_sft": {"cap": 25000},
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1": {
        "cap": 15000,
        "user": "responses_create_params.input",
        "free": "expected_action.content",
    },
    "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1": {
        "cap": 8000,
        "user": "responses_create_params.input",
        "free": "expected_answer",
    },
    # --- safety / guardrail / red-team ---
    "Lakera/mosscap_prompt_injection": {"cap": 30000},
    "yueliu1999/GuardReasonerTrain": {
        "cap": 20000,
        "streams": [
            (None, "WildGuardTrainR"),
            (None, "AegisTrainR"),
            (None, "BeaverTailsTrainR"),
            (None, "ToxicChatTrainR"),
        ],
        "user": "input",  # holds "Human user: ... AI assistant: ..." dialogue
        "free": "output",
    },
    "nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0": {
        "cap": 8000,
        "streams": [
            ("safety", "with_defense"),
            ("safety", "without_defense"),
            ("security", "with_defense"),
            ("security", "without_defense"),
        ],
        "user": "attack_snapshot.attack.injection_string",
        "free": "@aiq_trace",
    },
    "WNT3D/Ultimate-Offensive-Red-Team": {"cap": 8000},
    # --- instruction / SFT ---
    "HuggingFaceTB/smoltalk": {"cap": 15000, "streams": [("all", "train")]},
    "WizardLMTeam/WizardLM_evol_instruct_V2_196k": {"cap": 10000},
    "OpenCoder-LLM/opc-sft-stage2": {
        "cap": 12000,
        "streams": [
            ("educational_instruct", "train"),
            ("evol_instruct", "train"),
            ("mceval_instruct", "train"),
            ("package_instruct", "train"),
        ],
    },
    "theblackcat102/evol-codealpaca-v1": {"cap": 8000},
    # --- cyber ---
    "ethz-spylab/ctf-satml24": {
        "cap": 8000,
        "streams": [("interaction_chats", "attack"), ("defense", "valid")],
    },
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset": {"cap": 5000},
}

# chat-list fields flattened by role; `history` covers ctf interaction_chats
CHAT_FIELDS = (
    "messages",
    "conversation",
    "conversations",
    "dialogue",
    "dialog",
    "chat",
    "turns",
    "history",
)

USER_ROLES = {"user", "human", "client", "attacker", "customer"}
FREE_ROLES = {"assistant", "gpt", "model", "defender", "tool", "function"}
SYSTEM_ROLES = {"system"}

# fallback field maps used when no explicit hint / chat list matches
USER_FIELDS = (
    "instruction",
    "prompt",
    "query",
    "question",
    "jailbreak_query",
    "Goal",
    "Prompt",
    "input",
)
FREE_FIELDS = (
    "output",
    "answer",
    "response",
    "completion",
    "assistant",
    "raw_answer",
    "expected_answer",
    "Behavior",
    "rationale",
    "analysis",
    "label",
    "code",
)

ID_FIELDS = (
    "id",
    "_id",
    "example_id",
    "sample_id",
    "idx",
    "index",
    "uuid",
    "trajectory_id",
    "trace_id",
    "seq_id",
    "submission_id",
)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _to_str(v) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def path_get(obj, dotted: str):
    """Resolve a dotted path like `responses_create_params.input` into obj."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def chat_split(msgs) -> tuple[str, str] | None:
    """Split a chat list (role/content or sharegpt from/value) into
    (user_message, free_response)."""
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], dict):
        return None
    user_parts: list[str] = []
    free_parts: list[str] = []
    sys_parts: list[str] = []
    has_roles = any("role" in m or "from" in m for m in msgs if isinstance(m, dict))
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if has_roles:
            role = str(m.get("role") or m.get("from") or "").strip().lower()
            if not role:
                role = "user" if i % 2 == 0 else "assistant"
        else:
            role = "user" if i % 2 == 0 else "assistant"
        content = m.get("content")
        if content is None and "value" in m:
            content = m["value"]
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        if role in FREE_ROLES:
            free_parts.append(text)
        elif role in SYSTEM_ROLES:
            # system prompts are context, not model output: fold into the user side
            sys_parts.append(text)
        else:
            user_parts.append(text)
    user = "\n".join(sys_parts + user_parts).strip()
    free = "\n".join(free_parts).strip()
    # a chat with only assistant turns: keep the whole thing as free_response
    if not user and free:
        user, free = "", free
    if not user and not free:
        return None
    return (user, free)


def aiq_trace_free(rec: dict) -> str:
    """Scan the AIQ `trace` list for assistant output strings."""
    trace = rec.get("trace")
    if not isinstance(trace, list):
        return ""
    outs = []
    for ev in trace:
        if not isinstance(ev, dict):
            continue
        # attributes may be nested: {"attributes": {"output": {"value": ...}}}
        for key in ("attributes", "data", "output"):
            v = ev.get(key)
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if "output" in k2.lower() or "response" in k2.lower():
                        if isinstance(v2, dict) and "value" in v2:
                            v2 = v2["value"]
                        if isinstance(v2, str) and v2.strip():
                            outs.append(v2.strip())
        # top-level assistant message shapes
        for key in ("output", "response", "assistant"):
            v = ev.get(key)
            if isinstance(v, dict):
                v = v.get("content", v.get("value"))
            if isinstance(v, str) and v.strip():
                outs.append(v.strip())
    return "\n".join(dict.fromkeys(outs)).strip()  # dedupe, keep order


def extract_pair(rec: dict, spec: dict) -> tuple[str, str] | None:
    """Normalize one record to (user_message, free_response)."""
    if not isinstance(rec, dict):
        return None

    user: str | None = None
    free: str | None = None

    # 1. explicit per-dataset hints (dotted paths / @aiq_trace)
    if spec.get("user"):
        v = path_get(rec, spec["user"])
        if isinstance(v, list):
            pair = chat_split(v)
            if pair:
                user, free = pair
                if spec.get("free") and not free:
                    fv = path_get(rec, spec["free"])
                    if isinstance(fv, str) and fv.strip():
                        free = fv.strip()
                return (user, free)
            return None
        if isinstance(v, str) and v.strip():
            user = v.strip()
    if spec.get("free") == "@aiq_trace":
        free = aiq_trace_free(rec) or None
        if user and not free:
            free = ""
    elif spec.get("free"):
        fv = path_get(rec, spec["free"])
        if isinstance(fv, list):
            _, f2 = chat_split(fv) or (None, None)
            if f2:
                free = f2
        elif isinstance(fv, str) and fv.strip():
            free = fv.strip()
    if user is not None:
        return (user, free if free is not None else "")

    # 2. chat-list fields
    for key in CHAT_FIELDS:
        v = rec.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            pair = chat_split(v)
            if pair:
                return pair

    # 3. Trendyol-style system/user/assistant
    sys_, usr, ast = rec.get("system"), rec.get("user"), rec.get("assistant")
    if isinstance(usr, str) and usr.strip() and isinstance(ast, str) and ast.strip():
        user = (
            f"System: {sys_.strip()}\n\nUser: {usr.strip()}"
            if isinstance(sys_, str) and sys_.strip()
            else usr.strip()
        )
        return (user, ast.strip())

    # 4. paired instruction/prompt + output/answer/response
    for uk in USER_FIELDS:
        uv = rec.get(uk)
        if not isinstance(uv, str) or not uv.strip():
            continue
        user = uv.strip()
        # WNT3D: append `input` payload to the instruction
        if (
            uk == "instruction"
            and isinstance(rec.get("input"), str)
            and rec["input"].strip()
        ):
            user = f"{user}\n{rec['input'].strip()}"
        for fk in FREE_FIELDS:
            fv = rec.get(fk)
            if isinstance(fv, str) and fv.strip():
                free = fv.strip()
                break
        if free:
            return (user, free)
        break
    if user is not None:
        return (user, "")

    # 5. generic fallback: first string column -> user, rest joined -> free
    strings = [
        (k, str(v).strip()) for k, v in rec.items() if isinstance(v, str) and v.strip()
    ]
    if strings:
        user = strings[0][1]
        free = "\n".join(s for _, s in strings[1:])
        return (user, free)

    return None


def extract_id(rec: dict, user: str, free: str) -> str:
    for key in ID_FIELDS:
        v = rec.get(key)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return hashlib.sha1(f"{user}\x00{free}".encode("utf-8")).hexdigest()


def stream_records(dataset_id: str, cache_dir: str, config, split):
    """Yield records from a dataset, STREAMING. `config`/`split` may be None
    (default config / "train" with first-split fallback)."""
    from datasets import load_dataset

    kwargs = dict(streaming=True, cache_dir=cache_dir)
    if config:
        kwargs["name"] = config

    def _load(**kw):
        # trust_remote_code was removed in newer datasets versions; older ones
        # need it for custom loader scripts. Try with, fall back without.
        try:
            return load_dataset(
                dataset_id, split=split or "train", trust_remote_code=True, **kw
            )
        except TypeError:
            return load_dataset(dataset_id, split=split or "train", **kw)

    ds = _load(**kwargs)
    if isinstance(ds, dict):  # IterableDatasetDict -> first available split
        ds = ds[list(ds.keys())[0]]
    yield from ds


def est_tokens(user: str, free: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return (len(user) + len(free)) // 4


def upload_file(hf_bin: str, src: Path, bucket_uri: str) -> None:
    """Upload one file to a bucket via the hf CLI (its own auth)."""
    proc = subprocess.run(
        [hf_bin, "buckets", "cp", str(src), bucket_uri],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hf buckets cp failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    print(f"  [{_ts()}] uploaded {src.name} -> {bucket_uri}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--staging",
        default="/tmp/corpus-v3-staging",
        help="staging dir for per-dataset jsonl (default: /tmp/corpus-v3-staging)",
    )
    ap.add_argument(
        "--bucket",
        default="PeetPedro/quantal-corpus-v3",
        help="destination bucket (default: PeetPedro/quantal-corpus-v3)",
    )
    ap.add_argument(
        "--bucket-prefix",
        default="data",
        help="prefix inside the bucket (default: data)",
    )
    ap.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="safety override: cap every dataset at min(cap, N)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache-dir",
        default="/tmp/hf-datasets-cache",
        help="ephemeral datasets/hub cache dir (default: /tmp/hf-datasets-cache)",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="stage locally but do NOT upload to the bucket",
    )
    ap.add_argument(
        "--stream-chunk",
        type=int,
        default=500,
        help="samples buffered before flushing a dataset jsonl",
    )
    args = ap.parse_args()

    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print(
            f"[{_ts()}] note: HF_HUB_DISABLE_XET=1 (xet chunk caches off)", flush=True
        )

    try:
        import datasets  # noqa: F401  (deferred so --help works without it)
    except ImportError:
        print(
            "FATAL: the `datasets` library is required (streaming load). "
            "Run with:  uv run --with datasets --with pyarrow python scripts/build_corpus_v3.py",
            file=sys.stderr,
        )
        return 1

    hf_bin = shutil.which("hf")
    if hf_bin is None:
        print(
            "WARNING: `hf` CLI not found on PATH — upload will be skipped "
            "(run pip install -U 'hf-hub' and log in with `hf auth login`)",
            file=sys.stderr,
        )
        args.no_upload = True
    if hf_bin is None and not args.no_upload:
        args.no_upload = True
    if not args.no_upload:
        assert hf_bin is not None
        # sanity: verify bucket access BEFORE burning time on the build
        proc = subprocess.run(
            [hf_bin, "buckets", "list", args.bucket], capture_output=True, text=True
        )
        if proc.returncode != 0:
            print(
                f"WARNING: cannot list bucket {args.bucket}: "
                f"{proc.stderr.strip() or proc.stdout.strip()} — uploading will be skipped",
                file=sys.stderr,
            )
            args.no_upload = True

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    stats: dict[str, dict] = {}
    total_taken = 0
    total_tokens = 0
    raw_extracted = 0
    dedup_skipped_total = 0

    print(
        f"[{_ts()}] v3 corpus build start — {len(DATASETS)} datasets, seed={args.seed}",
        flush=True,
    )

    for i, (ds_id, spec) in enumerate(DATASETS.items(), 1):
        cap = spec["cap"]
        if args.max_per_dataset is not None:
            cap = min(cap, args.max_per_dataset)
        streams = spec.get("streams", [(None, None)])
        n_seen = 0
        n_taken = 0
        n_dedup = 0
        n_tokens = 0
        taken: list[dict] = []
        errors: list[str] = []
        t0 = datetime.now()

        for cfg, split in streams:
            if n_taken >= cap:
                break
            try:
                for rec in stream_records(ds_id, args.cache_dir, cfg, split):
                    n_seen += 1
                    if n_taken >= cap:
                        break
                    pair = extract_pair(rec, spec)
                    if pair is None:
                        continue
                    user, free = pair
                    raw_extracted += 1
                    uid = extract_id(rec, user, free)
                    h = hashlib.sha1(f"{user}\x00{free}".encode("utf-8")).hexdigest()
                    if h in seen_hashes or (uid in seen_ids):
                        n_dedup += 1
                        dedup_skipped_total += 1
                        continue
                    seen_hashes.add(h)
                    seen_ids.add(uid)
                    taken.append(
                        {"id": uid, "user_message": user, "free_response": free}
                    )
                    n_taken += 1
                    n_tokens += est_tokens(user, free)
            except Exception as e:  # one bad stream must not kill the dataset
                errors.append(
                    f"stream {cfg or '<default>'}/{split or '<default>'}: {type(e).__name__}: {e}"
                )
                print(
                    f"  [{_ts()}] {ds_id} [{cfg or '<default>'}/{split or '<default>'}] "
                    f"ERROR {type(e).__name__}: {e}",
                    flush=True,
                )

        if n_taken < cap and errors and not taken:
            print(
                f"  [{_ts()}] {ds_id}: FAILED — {len(errors)} stream error(s), 0 samples",
                flush=True,
            )

        # deterministic per-dataset shuffle (seed)
        rng.shuffle(taken)
        fname = f"{ds_id.replace('/', '__')}.jsonl"
        fpath = staging / fname
        with open(fpath, "w", encoding="utf-8") as fh:
            buf = []
            for row in taken:
                buf.append(json.dumps(row, ensure_ascii=False))
                if len(buf) >= args.stream_chunk:
                    fh.write("\n".join(buf) + "\n")
                    buf = []
            if buf:
                fh.write("\n".join(buf) + "\n")

        elapsed = (datetime.now() - t0).total_seconds()
        stats[ds_id] = {
            "cap": cap,
            "seen": n_seen,
            "taken": n_taken,
            "dedup_skipped": n_dedup,
            "est_tokens": n_tokens,
            "file": fname,
            "bytes": fpath.stat().st_size if fpath.exists() else 0,
            "elapsed_s": round(elapsed, 1),
            "stream_errors": errors,
        }
        total_taken += n_taken
        total_tokens += n_tokens
        print(
            f"[{_ts()}] {ds_id}: seen={n_seen} taken={n_taken}/{cap} "
            f"dedup={n_dedup} est_tokens={n_tokens} file={fname} "
            f"({round(fpath.stat().st_size / 1e6, 2)}MB, {elapsed:.0f}s)",
            flush=True,
        )

    # -- report.json in staging
    report = {
        "builder": "build_corpus_v3.py",
        "seed": args.seed,
        "schema": ["id", "user_message", "free_response"],
        "datasets": stats,
        "datasets_ok": sum(1 for s in stats.values() if s["taken"] > 0),
        "datasets_attempted": len(DATASETS),
        "total_taken": total_taken,
        "raw_extracted": raw_extracted,
        "total_dedup_skipped": dedup_skipped_total,
        "total_est_tokens": total_tokens,
        "staging": str(staging),
    }
    report_path = staging / "build-report-v3.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    total_bytes = sum(p.stat().st_size for p in staging.iterdir() if p.is_file())
    print(
        f"\n[{_ts()}] datasets OK: {report['datasets_ok']}/{len(DATASETS)}", flush=True
    )
    print(
        f"[{_ts()}] total samples: {total_taken} (raw extracted {raw_extracted}, "
        f"dedup skipped {dedup_skipped_total})",
        flush=True,
    )
    print(f"[{_ts()}] total est tokens: {total_tokens}", flush=True)
    print(f"[{_ts()}] staging: {staging} ({round(total_bytes / 1e6, 2)}MB)", flush=True)

    # -- upload
    if args.no_upload:
        print(
            f"[{_ts()}] UPLOAD SKIPPED (--no-upload). Staged files ready in {staging}.",
            flush=True,
        )
        return 0

    dest_prefix = f"hf://buckets/{args.bucket}/{args.bucket_prefix}"
    print(
        f"\n[{_ts()}] uploading {len(list(staging.iterdir()))} files -> {dest_prefix}/",
        flush=True,
    )
    assert hf_bin is not None  # guaranteed: upload only runs when the CLI exists
    upload_failures = 0
    for fpath in sorted(staging.iterdir()):
        if not fpath.is_file():
            continue
        try:
            upload_file(hf_bin, fpath, f"{dest_prefix}/{fpath.name}")
        except Exception as e:
            upload_failures += 1
            print(f"  [{_ts()}] upload FAILED for {fpath.name}: {e}", flush=True)
    if upload_failures:
        print(
            f"[{_ts()}] WARNING: {upload_failures} file(s) failed to upload — "
            f"retry with `hf buckets sync {staging} {dest_prefix}`",
            flush=True,
        )
        return 2
    print(
        f"[{_ts()}] CORPUS_V3_DONE n={total_taken} datasets={report['datasets_ok']} "
        f"bucket={args.bucket}/{args.bucket_prefix}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
