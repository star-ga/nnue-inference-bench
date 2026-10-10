#!/usr/bin/env python3
"""CPU reference smoke test (CI): model loads, trace matches the pinned checksum."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_cpu import compute_checksum, forward_batch, generate_positions, load_nknn, pack_inputs  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "nikola_d12v2_gold.nknn"
MODEL_SHA256 = "041BC71A527B830E54A83664BD7B2C4414A9F047348943DF2CE3359B49142E3C"
EXPECTED_CHECKSUM = 0xE67A3DAE  # 10 positions, seed 42


def main() -> int:
    model = load_nknn(MODEL)
    if model.sha256 != MODEL_SHA256:
        print(f"FAIL: model SHA-256 {model.sha256} != {MODEL_SHA256}")
        return 1

    trace = forward_batch(model, pack_inputs(generate_positions(10, 42)))
    checksum = compute_checksum(trace)
    print(f"checksum 0x{checksum:08X} (expected 0x{EXPECTED_CHECKSUM:08X})")

    # Guard against a degenerate check: the trace must actually vary by position.
    distinct = len(np.unique(trace, axis=0))
    if distinct != len(trace):
        print(f"FAIL: only {distinct} distinct trace rows for {len(trace)} positions")
        return 1
    if checksum != EXPECTED_CHECKSUM:
        print("FAIL: checksum mismatch")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
