#!/usr/bin/env python3
"""Streaming sentinel scan over the shipped corpus part dirs.

Scans data/part0 and data/part1 on HF without downloading everything at
once: fetches the paginated file list, then streams each file through curl
(pipe, no disk) and greps for [PHONE]/[EMAIL]/[IP] sentinels. Reports the
per-repo (jsonl file) hit counts and a row-level tally, matching the
deterministic-sample numbers P quoted (PHONE 1525 / EMAIL 196 / IP 35 in
the 60-file sample).

Usage: uv run --script scan_corpus_sentinels.py [part0|part1|flat|all]
"""

# /// script
# dependencies = []
# ///

import json
import re
import subprocess
import sys
import time
import urllib.request

DATASET = "PeetPedro/cogitoergosumma-corpus"
SENTINELS = ("[PHONE]", "[EMAIL]", "[IP]")

TREE_BASE = f"https://huggingface.co/api/datasets/{DATASET}/tree/main"


def tree_files(sub: str) -> list[str]:
    """Cursor-paginated file list under data/<sub>/ (sub without leading data/)."""
    files: list[str] = []
    url = f"{TREE_BASE}/data%2F{sub}?limit=1000"
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "sentinel-scan"})
        with urllib.request.urlopen(req, timeout=30) as r:
            entries = json.load(r)
            link = r.headers.get("Link", "")
        m = re.search(r"<([^>]+)>; rel=\"next\"", link)
        url = m.group(1) if m else None
        for e in entries:
            if e.get("type") == "file":
                files.append(e["path"])
    return files


def scan_file(path: str) -> dict[str, int]:
    """Stream one jsonl through curl and count sentinels per line."""
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}"
    hits = {"[PHONE]": 0, "[EMAIL]": 0, "[IP]": 0}
    try:
        proc = subprocess.Popen(
            ["curl", "-sL", "--max-time", "120", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            for s in SENTINELS:
                if s in line:
                    hits[s] += 1
        proc.wait(timeout=130)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ {path}: {e}", flush=True)
    return hits


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "part0"
    subs = {
        "part0": ["part0"],
        "part1": ["part1"],
        "flat": [""],
        "all": ["", "part0", "part1"],
    }[target]
    per_repo: dict[str, dict[str, int]] = {}
    total = {"[PHONE]": 0, "[EMAIL]": 0, "[IP]": 0}
    t0 = time.time()
    for sub in subs:
        label = sub if sub else "flat"
        files = tree_files(sub)
        print(f"[{label}] {len(files)} files to scan", flush=True)
        for i, path in enumerate(files):
            rel = path.removeprefix(f"data/{sub + '/' if sub else ''}")
            hits = scan_file(path)
            if any(hits.values()):
                per_repo[rel] = hits
                for s in SENTINELS:
                    total[s] += hits[s]
            if (i + 1) % 250 == 0:
                print(
                    f"[{label}] {i+1}/{len(files)} scanned, "
                    f"{len(per_repo)} repos hit, {time.time()-t0:.0f}s",
                    flush=True,
                )
    print("\n=== SCAN RESULT ===", flush=True)
    print(
        f"total sentinels: PHONE {total['[PHONE]']} | EMAIL {total['[EMAIL]']} | "
        f"IP {total['[IP]']}",
        flush=True,
    )
    print(f"repos with hits: {len(per_repo)}", flush=True)
    top = sorted(per_repo.items(), key=lambda kv: kv[1]["[PHONE]"], reverse=True)[:10]
    for name, h in top:
        print(f"  {name}: {h}", flush=True)
    with open(f"/tmp/sentinel-scan-{target}.json", "w") as f:
        json.dump({"total": total, "per_repo": per_repo}, f, indent=1)
    print(f"saved /tmp/sentinel-scan-{target}.json", flush=True)


if __name__ == "__main__":
    main()
