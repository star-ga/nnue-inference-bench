#!/usr/bin/env python3
"""
Runs every NNUE implementation on the same positions, checks each one's
per-position trace against the Python reference bit for bit, and writes one
JSON result file.

    python tools/run_benchmark.py                      # every implementation
    python tools/run_benchmark.py --impl python,cpp,simd --require python,cpp,simd

Implementations:
    python  NumPy reference (tools/infer_cpu.py)
    cpp     C++ scalar      (native-cpp, bench_cpu)
    simd    C++ AVX2        (native-cpp, bench_simd)
    mind    MIND CPU        (mind/nnue.mind via mindc --emit-shared)
    cuda    CUDA GPU        (native-cuda)

An implementation whose toolchain or hardware is missing is reported as
"unavailable" with the reason; one whose trace differs from the reference is
reported as "mismatch", one that crashes as "error", and either makes the run fail.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_cpu import (  # noqa: E402
    TRACE_WIDTH, compute_checksum, forward_batch, generate_positions, load_nknn,
    pack_inputs, write_inputs,
)

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "nikola_d12v2_gold.nknn"
WORK = REPO / "build" / "work"
EXE = ".exe" if os.name == "nt" else ""
CPP_BINARIES = {
    "cpp": ("C++ scalar", [REPO / "native-cpp" / "build" / f"bench_cpu{EXE}",
                           REPO / "native-cpp" / "build" / "Release" / f"bench_cpu{EXE}"]),
    "simd": ("C++ AVX2", [REPO / "native-cpp" / "build" / f"bench_simd{EXE}",
                          REPO / "native-cpp" / "build" / "Release" / f"bench_simd{EXE}"]),
}
ALL_IMPLS = ["python", "cpp", "simd", "mind", "cuda"]


class Unavailable(Exception):
    """The implementation cannot run here (missing toolchain, build or hardware)."""


def _version(cmd: List[str], contains: str = "") -> Optional[str]:
    """First output line of `cmd` (or the first containing `contains`); None if absent."""
    exe = cmd[0] if Path(cmd[0]).is_file() else shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in (out.stdout or out.stderr).splitlines() if contains in ln]
    return lines[0] if lines else None


def environment(cuda_compiler: Optional[str]) -> Dict[str, Optional[str]]:
    """Software versions only; hardware models are deliberately not recorded."""
    return {
        "os": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cxx": _version(["c++", "--version"]),
        "cmake": _version(["cmake", "--version"]),
        "nvcc": _version([cuda_compiler or "nvcc", "--version"], contains="release"),
        "mindc": _version([os.environ.get("MINDC") or "mindc", "--version"]),
    }


def summarize(times_ms: List[float], batch: int) -> Dict[str, float]:
    """Throughput uses the fastest batch: on a shared machine the minimum is the
    least disturbed measurement; p50/p95/mean are kept for comparison."""
    ordered = sorted(times_ms)
    best = ordered[0]
    return {
        "min_ms": round(best, 3),
        "p50_ms": round(ordered[len(ordered) // 2], 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
        "throughput_pos_per_s": round(batch / (best / 1000.0), 1),
    }


def time_calls(fn: Callable[[], object], warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def run_python(ctx: Dict) -> Dict:
    model, packed = ctx["model"], ctx["packed"]
    times = time_calls(lambda: forward_batch(model, packed), ctx["warmup"], ctx["iters"])
    return {"trace": forward_batch(model, packed), "times_ms": times,
            "timing_scope": "batched NumPy forward pass, inputs already packed in memory"}


def run_cpp(ctx: Dict, key: str) -> Dict:
    _, candidates = CPP_BINARIES[key]
    exe = next((p for p in candidates if p.exists()), None)
    if exe is None:
        raise Unavailable("not built: cmake -S native-cpp -B native-cpp/build && cmake --build native-cpp/build")
    out_path = WORK / f"trace_{key}.bin"
    out_path.unlink(missing_ok=True)
    proc = subprocess.run(
        [str(exe), "--model", str(MODEL), "--inputs", str(ctx["inputs_path"]), "--out", str(out_path),
         "--warmup", str(ctx["warmup"]), "--iters", str(ctx["iters"])],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{exe.name} exited {proc.returncode}: {proc.stderr.strip()}")
    report = json.loads(proc.stdout)
    trace = np.fromfile(out_path, dtype="<f4").reshape(-1, TRACE_WIDTH)
    return {"trace": trace, "times_ms": report["times_ms"],
            "timing_scope": "per-position forward pass in C++, inputs already in memory"}


def run_mind(ctx: Dict) -> Dict:
    import infer_mind
    try:
        lib = infer_mind.build()
    except infer_mind.MindUnavailable as exc:
        raise Unavailable(str(exc)) from exc
    mind = infer_mind.MindInference(ctx["model"], lib)
    mind.load_inputs(ctx["packed"])
    times = time_calls(mind.run_all, ctx["warmup"], ctx["iters"])
    trace = mind.trace()
    total = mind.run_all()
    if total != infer_mind.expected_total(trace):
        raise RuntimeError("MIND batch total disagrees with its per-value trace")
    return {"trace": trace, "times_ms": times,
            "timing_scope": "one MIND call per batch (all positions, all outputs), via ctypes"}


def run_cuda(ctx: Dict) -> Dict:
    import infer_gpu
    status = infer_gpu.get_gpu_status()
    if not status["available"]:
        raise Unavailable(status["blocked_reason"])
    gpu = infer_gpu.GPUInference(ctx["model"], ctx["packed"].shape[0])
    try:
        times = time_calls(lambda: gpu.forward(ctx["packed"]), ctx["warmup"], ctx["iters"])
        trace = gpu.forward(ctx["packed"])
    finally:
        gpu.close()
    return {"trace": trace, "times_ms": times,
            "timing_scope": "host-to-device input copy + kernels + device-to-host trace copy"}


def _cuda_compiler() -> Optional[str]:
    """The nvcc recorded in the CUDA build's CMake cache, if any."""
    cache = REPO / "native-cuda" / "build" / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_CUDA_COMPILER:"):
            return line.split("=", 1)[1].strip() or None
    return None


RUNNERS: Dict[str, Callable[[Dict], Dict]] = {
    "python": run_python,
    "cpp": lambda ctx: run_cpp(ctx, "cpp"),
    "simd": lambda ctx: run_cpp(ctx, "simd"),
    "mind": run_mind,
    "cuda": run_cuda,
}
NAMES = {"python": ("Python reference", "CPU"), "cpp": ("C++ scalar", "CPU"),
         "simd": ("C++ AVX2", "CPU"), "mind": ("MIND", "CPU"), "cuda": ("CUDA", "GPU")}


def evaluate(key: str, ctx: Dict) -> Dict:
    name, device = NAMES[key]
    row: Dict[str, object] = {"id": key, "implementation": name, "device": device}
    print(f"[{key}] running ...", flush=True)
    try:
        out = RUNNERS[key](ctx)
    except Unavailable as exc:
        row.update(status="unavailable", reason=str(exc))
        print(f"[{key}] unavailable: {exc}")
        return row
    except Exception as exc:  # a path that crashes must fail the run, not abort it
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        print(f"[{key}] error: {row['reason']}")
        return row
    trace = np.ascontiguousarray(out["trace"], dtype=np.float32)
    ref = ctx["reference"]
    same_shape = trace.shape == ref.shape
    identical = same_shape and np.array_equal(trace.view(np.uint32), ref.view(np.uint32))
    row.update(summarize(out["times_ms"], ctx["batch"]))
    row.update(
        status="verified" if identical else "mismatch",
        checksum=f"0x{compute_checksum(trace):08X}",
        bit_identical=bool(identical),
        max_abs_diff=float(np.abs(trace - ref).max()) if same_shape else None,
        timing_scope=out["timing_scope"],
    )
    print(f"[{key}] {row['status']}: {row['throughput_pos_per_s']:,.0f} pos/s, checksum {row['checksum']}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="all", help="comma list of " + ",".join(ALL_IMPLS) + " or 'all'")
    ap.add_argument("--require", default="python", help="comma list that must be available")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", "-o", default=str(WORK / "results.json"))
    ap.add_argument("--note", default="", help="free-text note stored with the results (hardware class, load)")
    args = ap.parse_args()

    impls = ALL_IMPLS if args.impl == "all" else [s.strip() for s in args.impl.split(",") if s.strip()]
    required = [s.strip() for s in args.require.split(",") if s.strip()]
    unknown = sorted(set(impls + required) - set(ALL_IMPLS))
    if unknown or not 1 <= args.batch <= 1024 or args.iters < 1 or args.warmup < 0:
        ap.error(f"bad arguments (unknown: {unknown}); batch must be 1..1024, iters >= 1")

    model = load_nknn(MODEL)
    packed = pack_inputs(generate_positions(args.batch, args.seed))
    WORK.mkdir(parents=True, exist_ok=True)
    inputs_path = WORK / f"inputs_seed{args.seed}_b{args.batch}.bin"
    write_inputs(inputs_path, packed)
    reference = forward_batch(model, packed)
    ctx = {"model": model, "packed": packed, "reference": reference, "inputs_path": inputs_path,
           "batch": args.batch, "warmup": args.warmup, "iters": args.iters}

    rows = [evaluate(key, ctx) for key in impls]
    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": shlex.join(["python", "tools/run_benchmark.py", *sys.argv[1:]]),
        "model": {"file": MODEL.name, "sha256": model.sha256, "bytes": model.file_size},
        "config": {"batch": args.batch, "warmup": args.warmup, "iters": args.iters, "seed": args.seed},
        "reference_checksum": f"0x{compute_checksum(reference):08X}",
        "environment": environment(_cuda_compiler()),
        "note": args.note,
        "results": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nreference checksum {result['reference_checksum']}; results written to {out_path}")

    failed = [r["id"] for r in rows if r["status"] in ("mismatch", "error")]
    missing = [r["id"] for r in rows if r["status"] == "unavailable" and r["id"] in required]
    missing += [k for k in required if k not in impls]
    if failed or missing:
        print(f"FAIL: mismatch-or-error={failed} required-but-unavailable={missing}")
        return 1
    print("PASS: every implementation that ran is bit-identical to the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
