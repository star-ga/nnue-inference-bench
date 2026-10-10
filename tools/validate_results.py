#!/usr/bin/env python3
"""
CI gate for a results JSON written by tools/run_benchmark.py.

Fails when:
  * the reference checksum differs from the pinned value for that
    batch/seed (the reference numerics changed, e.g. a NumPy or platform
    difference, or an edit to tools/infer_cpu.py), or
  * any implementation that ran is not bit-identical to the reference.

    python tools/validate_results.py results/latest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Pinned reference checksums: (batch, seed) -> checksum of the float32 trace.
PINNED = {
    (10, 42): "0xE67A3DAE",
    (1000, 42): "0x0613C9FD",
}


def validate(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data["config"]
    errors = []
    pinned = PINNED.get((cfg["batch"], cfg["seed"]))
    if pinned is None:
        print(f"note: no pinned checksum for batch={cfg['batch']} seed={cfg['seed']}")
    elif data["reference_checksum"] != pinned:
        errors.append(f"reference checksum {data['reference_checksum']} != pinned {pinned}")

    ran = 0
    for row in data["results"]:
        if row["status"] == "unavailable":
            print(f"  [skip] {row['implementation']}: {row['reason']}")
            continue
        ran += 1
        ok = row["status"] == "verified" and row["checksum"] == data["reference_checksum"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {row['implementation']}: {row['checksum']}")
        if not ok:
            errors.append(f"{row['implementation']} is not bit-identical to the reference")
    if ran == 0:
        errors.append("no implementation ran")

    for err in errors:
        print("ERROR:", err)
    return 1 if errors else 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    return validate(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
