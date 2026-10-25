#!/usr/bin/env python3
"""Pure-ASCII progress HUD for the cogito corpus scans.

Reads the v4 scan logs + part1 result JSON and renders a compact
ASCII progress-bar HUD. No unicode, no emoji, no network.

Usage: uv run --script cogito_hud.py
"""

# /// script
# dependencies = []
# ///

import json
import os
import re
import sys

PART0_LOG = "/tmp/sentinel-scan-part0-v4.log"
FLAT_LOG = "/tmp/sentinel-scan-flat-v4.log"
PART0_JSON = "/tmp/sentinel-scan-part0.json"
FLAT_JSON = "/tmp/sentinel-scan-flat.json"
PART1_JSON = "/tmp/sentinel-scan-part1.json"
BAR_W = 32


def parse_log(path: str):
    if not os.path.exists(path):
        return None
    for line in reversed(open(path, errors="ignore").read().splitlines()):
        m = re.search(r"(\d+)/(\d+) done, (\d+) hit, (\d+)s", line)
        if m:
            return {
                "done": int(m[1]),
                "total": int(m[2]),
                "hit": int(m[3]),
                "secs": int(m[4]),
            }
    return None


def parse_json(path: str, total_files: int):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {
        "done": total_files,
        "total": total_files,
        "hit": len(d.get("per_repo", {})),
        "secs": 0,
        "phones": d["total"].get("[PHONE]", 0),
        "emails": d["total"].get("[EMAIL]", 0),
        "ips": d["total"].get("[IP]", 0),
    }


def bar(frac: float) -> str:
    filled = int(round(frac * BAR_W))
    return "[" + "#" * filled + "-" * (BAR_W - filled) + "]"


def eta(secs: float, done: int, total: int) -> str:
    if done <= 0 or secs <= 0:
        return "n/a"
    remaining = secs / done * (total - done)
    if remaining > 3600:
        return f"{remaining/3600:.1f}h"
    return f"{remaining/60:.0f}m"


def row(name: str, d: dict, extra: str = "") -> str:
    if d is None:
        return f"{name:<6} [---- no data ----]"
    frac = d["done"] / d["total"]
    pct = f"{frac*100:5.1f}%"
    el = f"{d['secs']//60:3d}m"
    e = f"eta {eta(d['secs'], d['done'], d['total']):>5}"
    hits = f"hits {d.get('phones', d.get('hit', 0)):>6}"
    return (
        f"{name:<6} {bar(frac)} {pct}  {d['done']:>5}/{d['total']:<5} "
        f"{el} {e} {hits} {extra}"
    )


def main() -> None:
    p0 = parse_json(PART0_JSON, 9991) or parse_log(PART0_LOG)
    fl = parse_json(FLAT_JSON, 9988) or parse_log(FLAT_LOG)
    p1 = parse_json(PART1_JSON, 1878)

    parts = [p for p in (p0, fl, p1) if p]
    tot_done = sum(p["done"] for p in parts)
    tot_all = sum(p["total"] for p in parts)
    tot_phones = p1["phones"] if p1 else 0
    if p0:
        tot_phones += p0.get("phones", 0)
    if fl:
        tot_phones += fl.get("phones", 0)

    print("=" * 58)
    print(" cogito sentinel scan HUD   (pure ascii, refresh 15m)")
    print("=" * 58)
    print(row("part0", p0))
    print(row("flat ", fl))
    print(
        row(
            "part1",
            p1,
            f"| P {p1['phones']:,} E {p1['emails']:,} I {p1['ips']:,}" if p1 else "",
        )
    )
    print("-" * 58)
    print(
        f" TOTAL {bar(tot_done/tot_all)} {tot_done/tot_all*100:5.1f}%  "
        f"{tot_done:,}/{tot_all:,}  PHONE {tot_phones:,}"
    )
    print("=" * 58)


if __name__ == "__main__":
    main()
