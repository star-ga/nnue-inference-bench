#!/usr/bin/env python3
"""
MIND CPU path: builds mind/nnue.mind into a shared library with `mindc` and
drives it through the static-shape tensor C ABI (ctypes).

Each 1-D tensor parameter is passed as the unpacked memref descriptor
(alloc_ptr, aligned_ptr, offset, size, stride).

Requires a MIND 0.10.4 compiler (`mindc` on PATH or in $MINDC) built with
the MLIR backend, plus the LLVM/MLIR command-line tools it invokes.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_cpu import INPUT_WIDTH, TRACE_WIDTH, NKNNModel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MIND_SRC = REPO / "mind" / "nnue.mind"
MIND_LIB = REPO / "mind" / "build" / ("nnue_mind.dll" if os.name == "nt" else "libnnue_mind.so")
MAX_POSITIONS = 1024          # static shape of the `feats` tensor
SUPPORTED_SERIES = "0.10."

_P = ctypes.c_void_p
_I = ctypes.c_longlong


class MindUnavailable(RuntimeError):
    """Raised when the MIND path cannot be built or loaded."""


def find_mindc() -> Optional[str]:
    override = os.environ.get("MINDC")
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("mindc")


def mindc_version(mindc: str) -> str:
    out = subprocess.run([mindc, "--version"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise MindUnavailable(f"`{mindc} --version` exited {out.returncode}")
    return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"


def build(force: bool = False) -> Path:
    """Compile mind/nnue.mind to a shared library (skipped when up to date)."""
    mindc = find_mindc()
    if mindc is None:
        raise MindUnavailable("mindc not found (install MIND 0.10.4 or set MINDC)")
    version = mindc_version(mindc)
    if SUPPORTED_SERIES not in version:
        raise MindUnavailable(f"need MIND {SUPPORTED_SERIES}x, found: {version}")
    if (not force and MIND_LIB.exists()
            and MIND_LIB.stat().st_mtime >= MIND_SRC.stat().st_mtime):
        return MIND_LIB
    MIND_LIB.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        [mindc, str(MIND_SRC), "--emit-shared", str(MIND_LIB)],
        capture_output=True, text=True, timeout=900,
    )
    if out.returncode != 0 or not MIND_LIB.exists():
        tail = (out.stderr or out.stdout).strip().splitlines()[-5:]
        raise MindUnavailable(f"mindc exited {out.returncode}: " + " | ".join(tail))
    return MIND_LIB


def _desc(arr: np.ndarray) -> List[object]:
    ptr = arr.ctypes.data_as(_P)
    return [ptr, ptr, 0, arr.shape[0], 1]


class MindInference:
    """NNUE forward pass executed by the MIND-compiled shared library."""

    def __init__(self, model: NKNNModel, lib_path: Optional[Path] = None):
        path = lib_path or build()
        self.lib = ctypes.CDLL(str(path))
        self.fn = self.lib.nnue_run
        f64 = np.float64
        # Keep every buffer referenced for the lifetime of the object.
        self.weights = [
            np.ascontiguousarray(model.W1.reshape(-1), dtype=f64),
            np.ascontiguousarray(model.B1, dtype=f64),
            np.ascontiguousarray(model.W2.reshape(-1), dtype=f64),
            np.ascontiguousarray(model.B2, dtype=f64),
            np.ascontiguousarray(model.W3.reshape(-1), dtype=f64),
            np.ascontiguousarray(model.B3, dtype=f64),
            np.ascontiguousarray(model.W4.reshape(-1), dtype=f64),
            np.ascontiguousarray(model.B4, dtype=f64),
            np.ascontiguousarray(model.W_wdl.reshape(-1), dtype=f64),
            np.ascontiguousarray(model.B_wdl, dtype=f64),
        ]
        self.feats = np.zeros(MAX_POSITIONS * INPUT_WIDTH, dtype=np.int64)
        self.scratch = [np.zeros(512, f64), np.zeros(32, f64), np.zeros(32, f64)]
        n_tensors = len(self.weights) + 1 + len(self.scratch)
        self.fn.argtypes = [_P, _P, _I, _I, _I] * n_tensors + [_I, _I, _I]
        self.fn.restype = ctypes.c_double
        self.prefix: List[object] = []
        for arr in self.weights + [self.feats] + self.scratch:
            self.prefix.extend(_desc(arr))
        self.count = 0

    def load_inputs(self, packed: np.ndarray) -> None:
        if packed.shape[0] > MAX_POSITIONS or packed.shape[1] != INPUT_WIDTH:
            raise ValueError(f"inputs must be [<= {MAX_POSITIONS}, {INPUT_WIDTH}]")
        self.feats[:] = 0
        self.feats[:packed.size] = packed.reshape(-1)
        self.count = packed.shape[0]

    def run_all(self) -> float:
        """Timed path: every position, returns the f64 sum of all trace values."""
        return self.fn(*self.prefix, 0, self.count, -1)

    def trace(self) -> np.ndarray:
        """Verification path: the float32 [N, 68] trace, one value per call."""
        out = np.zeros((self.count, TRACE_WIDTH), dtype=np.float32)
        for p in range(self.count):
            for h in range(TRACE_WIDTH):
                out[p, h] = self.fn(*self.prefix, p, 1, h)
        return out


def expected_total(trace: np.ndarray) -> float:
    """The f64 sum run_all() must return, in the kernel's accumulation order."""
    total = 0.0
    order = list(range(4, TRACE_WIDTH)) + list(range(4))
    for row in trace.astype(np.float64):
        for h in order:
            total += float(row[h])
    return total


if __name__ == "__main__":
    lib = build(force="--force" in sys.argv)
    print(f"built {lib.relative_to(REPO)} with {mindc_version(find_mindc() or 'mindc')}")
