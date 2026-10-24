#!/usr/bin/env python3
"""enumerate_standardgalactic.py — resumable full repo enumeration.

Fetches ALL of standardgalactic's public repos page by page (24k+), appending
each page to a JSONL so an interruption never loses progress. The final
non-fork list is written to /tmp/cogito/repos_nofork.json.
"""

from __future__ import annotations

import json
import subprocess
import time

OUT = "/tmp/cogito/repos_full.jsonl"
NOFORK = "/tmp/cogito/repos_nofork.json"
MAX_PAGES = 260  # 24,154 / 100 = 242 pages


def fetch_page(page: int) -> list | None:
    out = subprocess.run(
        ["gh", "api", f"users/standardgalactic/repos?per_page=100&page={page}"],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if out.returncode != 0:
        print(f"page {page}: gh error: {out.stderr[:120]}", flush=True)
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        print(f"page {page}: bad json", flush=True)
        return None


def main() -> None:
    import os

    seen_pages = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    seen_pages.add(json.loads(line)["_page"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
    print(f"resuming: {len(seen_pages)} pages already fetched", flush=True)

    with open(OUT, "a") as f:
        for page in range(1, MAX_PAGES + 1):
            if page in seen_pages:
                continue
            batch = fetch_page(page)
            if batch is None:
                time.sleep(3)
                continue
            if not batch:
                print(f"page {page}: empty → done", flush=True)
                break
            rec = {"_page": page, "repos": batch}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"page {page}: +{len(batch)} (cumulative ~{page*100})", flush=True)
            time.sleep(0.25)

    # build the final non-fork list
    all_repos = []
    with open(OUT) as f:
        for line in f:
            try:
                all_repos.extend(json.loads(line)["repos"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    notforks = [r for r in all_repos if not r.get("fork")]
    json.dump(notforks, open(NOFORK, "w"))
    tot = sum(r.get("size", 0) for r in notforks)
    print(
        f"\nTOTAL: {len(all_repos)} repos | non-fork: {len(notforks)} | {tot/1024/1024:.1f} GB"
    )
    print(f"non-fork >=100KB: {len([r for r in notforks if r.get('size',0)>=100])}")


if __name__ == "__main__":
    main()
