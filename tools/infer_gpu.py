#!/usr/bin/env python3
"""
GPU inference for NKNN v2 through the native CUDA kernels (ctypes).

Build the kernels first:
    cmake -S native-cuda -B native-cuda/build && cmake --build native-cuda/build

Every call reports failure explicitly; when no GPU or no compiled library is
present, get_gpu_status() returns the reason instead of raising.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_cpu import INPUT_WIDTH, OUTPUTS_PER_POSITION, TRACE_WIDTH, NKNNModel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIB_CANDIDATES = [
    REPO / "native-cuda" / "build" / "nnue_cuda.so",
    REPO / "native-cuda" / "build" / "nnue_cuda.dll",
    REPO / "native-cuda" / "build" / "Release" / "nnue_cuda.dll",
]
L1, L2, L3 = 256, 32, 32
API_VERSION = 2

_P = ctypes.c_void_p
_I = ctypes.c_int


class GPUError(RuntimeError):
    """A CUDA call returned a non-zero error code."""


def _find_library() -> Optional[Path]:
    return next((p for p in LIB_CANDIDATES if p.exists()), None)


def _setup(lib: ctypes.CDLL) -> None:
    lib.nnue_get_version.argtypes = []
    lib.nnue_get_version.restype = _I
    lib.nnue_init.argtypes = [_I]
    lib.nnue_sync.argtypes = []
    lib.nnue_malloc.argtypes = [ctypes.POINTER(_P), ctypes.c_size_t]
    lib.nnue_free.argtypes = [_P]
    lib.nnue_memcpy_h2d.argtypes = [_P, _P, ctypes.c_size_t]
    lib.nnue_memcpy_d2h.argtypes = [_P, _P, ctypes.c_size_t]
    lib.nnue_l1.argtypes = [_P, _P, _P, _P, _I]
    lib.nnue_dense.argtypes = [_P, _P, _P, _P, _P, _I, _I, _I]
    for name in ("nnue_init", "nnue_sync", "nnue_malloc", "nnue_free", "nnue_memcpy_h2d",
                 "nnue_memcpy_d2h", "nnue_l1", "nnue_dense"):
        getattr(lib, name).restype = _I


def get_gpu_status() -> Dict[str, Any]:
    """Returns {available, blocked_reason}; never raises."""
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if smi.returncode != 0:
            return {"available": False, "blocked_reason": "nvidia-smi failed (no NVIDIA GPU or driver)"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "blocked_reason": f"nvidia-smi unavailable: {exc}"}
    path = _find_library()
    if path is None:
        return {"available": False, "blocked_reason": "CUDA kernels not built (see native-cuda/README.md)"}
    try:
        lib = ctypes.CDLL(str(path))
        _setup(lib)
    except OSError as exc:
        return {"available": False, "blocked_reason": f"cannot load {path.name}: {exc}"}
    if lib.nnue_get_version() != API_VERSION:
        return {"available": False, "blocked_reason": f"{path.name} is stale; rebuild native-cuda"}
    err = lib.nnue_init(0)
    if err != 0:
        return {"available": False, "blocked_reason": f"cudaSetDevice failed (error {err})"}
    return {"available": True, "blocked_reason": None, "library": str(path.relative_to(REPO))}


class GPUInference:
    """Weights resident on the device; buffers sized for one batch and reused."""

    def __init__(self, model: NKNNModel, batch_size: int):
        path = _find_library()
        if path is None:
            raise GPUError("CUDA kernels not built")
        self.lib = ctypes.CDLL(str(path))
        _setup(self.lib)
        self._check(self.lib.nnue_init(0), "init")
        self.batch = batch_size
        self.ptrs: Dict[str, _P] = {}
        heads_w = np.concatenate([model.W4_q, model.W_wdl_q], axis=1).astype(np.int8)
        heads_b = np.concatenate([model.B4_q, model.B_wdl_q]).astype(np.int16)
        host = {
            "W1": np.ascontiguousarray(model.W1_q, dtype=np.int16),
            "B1": np.ascontiguousarray(model.B1_q, dtype=np.int16),
            "W2": np.ascontiguousarray(model.W2_q, dtype=np.int8),
            "B2": np.ascontiguousarray(model.B2_q, dtype=np.int16),
            "W3": np.ascontiguousarray(model.W3_q, dtype=np.int8),
            "B3": np.ascontiguousarray(model.B3_q, dtype=np.int16),
            "WH": np.ascontiguousarray(heads_w),
            "BH": np.ascontiguousarray(heads_b),
        }
        for name, arr in host.items():
            self._alloc(name, arr.nbytes)
            self._check(self.lib.nnue_memcpy_h2d(self.ptrs[name], arr.ctypes.data, arr.nbytes), f"upload {name}")
        b = batch_size
        for name, nbytes in (("rows", b * INPUT_WIDTH * 4), ("h1", b * 2 * L1 * 4),
                             ("pre2", b * L2 * 4), ("h2", b * L2 * 4), ("pre3", b * L3 * 4),
                             ("h3", b * L3 * 4), ("heads", b * OUTPUTS_PER_POSITION * 4)):
            self._alloc(name, nbytes)
        self._check(self.lib.nnue_sync(), "sync")

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            raise GPUError(f"CUDA {what} failed (error {err})")

    def _alloc(self, name: str, nbytes: int) -> None:
        ptr = _P()
        self._check(self.lib.nnue_malloc(ctypes.byref(ptr), nbytes), f"malloc {name}")
        self.ptrs[name] = ptr

    def forward(self, packed: np.ndarray) -> np.ndarray:
        """Uploads packed inputs, runs every layer, returns the float32 [N, 68] trace."""
        n = packed.shape[0]
        if n > self.batch or packed.shape[1] != INPUT_WIDTH:
            raise ValueError(f"inputs must be [<= {self.batch}, {INPUT_WIDTH}]")
        rows = np.ascontiguousarray(packed, dtype=np.int32)
        p, lib, chk = self.ptrs, self.lib, self._check
        chk(lib.nnue_memcpy_h2d(p["rows"], rows.ctypes.data, rows.nbytes), "upload rows")
        chk(lib.nnue_l1(p["W1"], p["B1"], p["rows"], p["h1"], n), "l1")
        chk(lib.nnue_dense(p["W2"], p["B2"], p["h1"], p["pre2"], p["h2"], n, 2 * L1, L2), "l2")
        chk(lib.nnue_dense(p["W3"], p["B3"], p["h2"], p["pre3"], p["h3"], n, L2, L3), "l3")
        chk(lib.nnue_dense(p["WH"], p["BH"], p["h3"], p["heads"], None, n, L3, OUTPUTS_PER_POSITION), "heads")
        heads = np.empty((n, OUTPUTS_PER_POSITION), np.float32)
        pre2 = np.empty((n, L2), np.float32)
        pre3 = np.empty((n, L3), np.float32)
        for arr, name in ((heads, "heads"), (pre2, "pre2"), (pre3, "pre3")):
            chk(lib.nnue_memcpy_d2h(arr.ctypes.data, p[name], arr.nbytes), f"download {name}")
        chk(lib.nnue_sync(), "sync")
        trace = np.concatenate([heads, pre2, pre3], axis=1)
        assert trace.shape[1] == TRACE_WIDTH
        return trace

    def close(self) -> None:
        for ptr in self.ptrs.values():
            if ptr.value:
                self.lib.nnue_free(ptr)
        self.ptrs = {}


if __name__ == "__main__":
    for key, value in get_gpu_status().items():
        print(f"{key}: {value}")
