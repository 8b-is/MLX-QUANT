#!/usr/bin/env python3
"""Periodic model-card updater for the cogito corpus.

Reads the live sentinel-scan result files (/tmp/sentinel-scan-*.json),
regenerates the "Scrubber audit" section of the README, and uploads it
to the HF dataset repo — but ONLY when the numbers actually changed, so
a cron loop can call this every N minutes without spamming commits.

Usage:
    uv run --script update_model_card.py            # check + upload if changed
    uv run --script update_model_card.py --force    # upload regardless
    uv run --script update_model_card.py --print    # print README, no upload

Result files (all optional; missing ones are reported as "pending"):
    /tmp/sentinel-scan-part0.json   part0 scan (exact counts, checkpointed)
    /tmp/sentinel-scan-part1.json   part1 scan (exact counts)
    /tmp/sentinel-scan-flat.json    flat scan (exact counts, checkpointed)
"""

# /// script
# dependencies = []
# ///

import json
import os
import subprocess
import sys

SCAN_FILES = {
    "part0": "/tmp/sentinel-scan-part0.json",
    "part1": "/tmp/sentinel-scan-part1.json",
    "flat": "/tmp/sentinel-scan-flat.json",
}
PROGRESS_FILES = {
    "part0": "/tmp/sentinel-scan-part0.progress",
    "flat": "/tmp/sentinel-scan-flat.progress",
}
README_TMPL = "/tmp/cogito-readme-template.md"
README_OUT = "/tmp/cogito-readme-live.md"
STATE = "/tmp/cogito-readme-last-numbers.json"
DATASET = "PeetPedro/cogitoergosumma-corpus"


def load_scans() -> dict:
    out = {}
    for name, path in SCAN_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            out[name] = {
                "total": d.get("total", {}),
                "per_repo": d.get("per_repo", {}),
                "done": len(d.get("per_repo", {})),
            }
        elif name in PROGRESS_FILES and os.path.exists(PROGRESS_FILES[name]):
            with open(PROGRESS_FILES[name]) as f:
                n = len(json.load(f))
            out[name] = {"total": {}, "per_repo": {}, "done": n}
    return out


def numbers_sig(scans: dict) -> str:
    parts = []
    for name in ("part0", "part1", "flat"):
        if name in scans and scans[name]["total"]:
            t = scans[name]["total"]
            parts.append(
                f"{name}:P{t.get('[PHONE]', 0)}"
                f"/E{t.get('[EMAIL]', 0)}/I{t.get('[IP]', 0)}"
                f"/R{len(scans[name]['per_repo'])}"
            )
        elif name in scans:
            parts.append(f"{name}:scanning({scans[name]['done']} files)")
        else:
            parts.append(f"{name}:pending")
    return "|".join(parts)


def build_audit_section(scans: dict) -> str:
    rows = []
    grand = {"[PHONE]": 0, "[EMAIL]": 0, "[IP]": 0}
    repos = set()
    for name in ("part0", "part1", "flat"):
        if name not in scans:
            rows.append(f"| {name} | _pending_ | — | — | — |")
            continue
        t = scans[name]["total"]
        if not t:
            rows.append(
                f"| {name} | _scanning ({scans[name]['done']} files done)_ | — | — | — |"
            )
            continue
        for s in grand:
            grand[s] += t.get(s, 0)
        repos |= set(scans[name]["per_repo"].keys())
        rows.append(
            f"| {name} | done | **{t.get('[PHONE]', 0):,}** | "
            f"{t.get('[EMAIL]', 0):,} | {t.get('[IP]', 0):,} |"
        )
    if grand["[PHONE]"] or grand["[EMAIL]"] or grand["[IP]"]:
        rows.append(
            f"| **total** | **{len(repos):,} affected files** | "
            f"**{grand['[PHONE]']:,}** | {grand['[EMAIL]']:,} | "
            f"{grand['[IP]']:,} |"
        )
    table = "\n".join(rows)

    top = []
    for name in ("part0", "part1", "flat"):
        if name in scans:
            pr = scans[name]["per_repo"]
            for f, h in sorted(
                pr.items(), key=lambda kv: kv[1].get("[PHONE]", 0), reverse=True
            )[:5]:
                top.append(f"  `{f}`: PHONE {h.get('[PHONE]', 0)}")
    top_s = "\n".join(top[:10]) if top else "  _still scanning…_"

    return f"""### Live corpus-wide scan (periodic)

Scans run rate-limited (128 KB/s per worker — bandwidth is second-class
while they run); the model card is updated periodically as checkpoints
land. Counts are exact sentinel occurrences.

| part | status | PHONE | EMAIL | IP |
|---|---|---|---|---|
{table}

Top affected files so far:

{top_s}

**Re-emit rule**: every affected file is re-fetched and re-scrubbed with
the fixed regex (commit `1a19de50`) — `id` is path-derived, so rows are
overwritten in place, never re-cloned. The go.sum pseudo-version digits
are restored from the module proxy."""


def build_readme(scans: dict) -> str:
    with open(README_TMPL) as f:
        tmpl = f.read()
    marker = "{{LIVE_SCAN_SECTION}}"
    if marker not in tmpl:
        raise SystemExit(f"template {README_TMPL} missing {marker} marker")
    return tmpl.replace(marker, build_audit_section(scans))


def main() -> None:
    force = "--force" in sys.argv
    print_only = "--print" in sys.argv
    scans = load_scans()
    sig = numbers_sig(scans)
    print(f"numbers: {sig}")

    if not force and not print_only and os.path.exists(STATE):
        with open(STATE) as f:
            if json.load(f).get("sig") == sig:
                print("unchanged — skip upload")
                return

    readme = build_readme(scans)
    with open(README_OUT, "w") as f:
        f.write(readme)
    print(f"built {README_OUT}")

    if print_only:
        return

    r = subprocess.run(
        ["hf", "upload", DATASET, README_OUT, "README.md", "--repo-type", "dataset"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        print("upload FAILED:", r.stderr[-300:])
        sys.exit(1)
    with open(STATE, "w") as f:
        json.dump({"sig": sig}, f)
    print("uploaded:", r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
