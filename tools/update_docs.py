#!/usr/bin/env python3
"""
Renders the README results table from a results JSON written by
tools/run_benchmark.py.

    python tools/update_docs.py                      # rewrite README from results/latest.json
    python tools/update_docs.py --check              # exit 1 if README is out of date
    python tools/update_docs.py --results other.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
DEFAULT_RESULTS = REPO / "results" / "latest.json"
START, END = "<!-- AUTO:BENCHMARK_START -->", "<!-- AUTO:BENCHMARK_END -->"


def render(data: Dict) -> str:
    rows: List[Dict] = data["results"]
    base = next((r for r in rows if r["id"] == "cpp" and r["status"] == "verified"), None)
    lines = [
        "| Implementation | Device | Best batch (ms) | p50 batch (ms) | Throughput (pos/s) "
        "| vs C++ scalar | Trace checksum | Bit-identical |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if r["status"] in ("unavailable", "error"):
            label = "not run" if r["status"] == "unavailable" else "ERROR"
            lines.append(f"| {r['implementation']} | {r['device']} | - | - | - | - | - | {label}: {r['reason']} |")
            continue
        speedup = (f"{r['throughput_pos_per_s'] / base['throughput_pos_per_s']:.2f}x"
                   if base else "-")
        lines.append(
            f"| {r['implementation']} | {r['device']} | {r['min_ms']:.3f} | {r['p50_ms']:.3f} | "
            f"{r['throughput_pos_per_s']:,.0f} | {speedup} | `{r['checksum']}` | "
            f"{'yes' if r['bit_identical'] else 'NO'} |"
        )
    cfg, env = data["config"], data["environment"]
    lines += [
        "",
        f"Reference checksum `{data['reference_checksum']}` (batch {cfg['batch']}, seed {cfg['seed']}, "
        f"{cfg['warmup']} warmup + {cfg['iters']} measured iterations). "
        f"Run {data['generated_utc'][:10]} with `{data['command']}`.",
        "",
        f"Environment: {env['os']}, Python {env['python']}, NumPy {env['numpy']}, "
        f"{env['cxx'] or 'no C++ compiler'}, {env['nvcc'] or 'no nvcc'}, "
        f"{env['mindc'] or 'no mindc'}. Throughput uses the fastest measured batch.",
    ]
    if data.get("note"):
        lines += ["", f"Note: {data['note'].rstrip('.')}."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--check", action="store_true", help="fail if README differs from the results")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        print(f"README.md has no {START} ... {END} block")
        return 1
    updated = pattern.sub(lambda _: f"{START}\n{render(data)}\n{END}", readme)

    if args.check:
        if updated != readme:
            print("README results table does not match", args.results)
            return 1
        print("README results table matches", args.results)
        return 0
    if updated != readme:
        README.write_text(updated, encoding="utf-8")
        print("README.md updated from", args.results)
    else:
        print("README.md already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
