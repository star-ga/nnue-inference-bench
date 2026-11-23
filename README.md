<p align="center">
  <img src="assets/nnue-inference-bench.svg" alt="NNUE Inference Bench logo" width="96" height="96">
</p>

<h1 align="center">NNUE Inference Bench</h1>

<p align="center">
  One NNUE network, five implementations (Python, C++ scalar, C++ AVX2, MIND, CUDA),
  checked bit for bit against each other and timed on the same positions.
</p>

## What it is

A small, reproducible benchmark for NNUE chess-evaluation inference on the
NKNN v2 format (`40960 -> 256x2 -> 32 -> 32 -> 1` plus a 3-way WDL head). Every
implementation evaluates the same 1,000 seeded positions and must produce a
per-position trace that is **bit-identical** to the NumPy reference before its
timing is reported.

| Path | Source | Build |
|---|---|---|
| Python reference | `tools/infer_cpu.py` | NumPy |
| C++ scalar | `native-cpp/nnue_cpu.cpp` | CMake, any C++17 compiler |
| C++ AVX2 | `native-cpp/nnue_cpu_simd.cpp` | CMake, x86-64 with AVX2 |
| MIND | `mind/nnue.mind` | `mindc --emit-shared`, called through the tensor C ABI |
| CUDA | `native-cuda/nnue_kernels.cu` | CMake + CUDA toolkit, NVIDIA GPU |

## Status

- All five paths build and run on Linux x86-64 and produce the same trace checksum.
- CI (Linux and Windows) builds the C++ paths and gates the Python, C++ scalar and
  C++ AVX2 paths on bit identity with a pinned reference checksum.
- The MIND and CUDA paths are run locally (they need `mindc` and an NVIDIA GPU).
- The MIND kernel needs a `mindc` built from the [MIND repository](https://github.com/star-ga/mind)
  `main` branch (tested at commit `5724a6ee`, version 0.10.2). The prebuilt
  v0.10.2 release binary predates the runtime-indexed tensor parameters it uses
  and stops with `missing type information ... while lowering binop`.

## Quick start

Requirements: Python 3.11+ with NumPy, CMake 3.24+, a C++17 compiler.
Optional: a MIND 0.10.x compiler (see above) with the LLVM/MLIR tools it calls,
and a CUDA toolkit with an NVIDIA GPU.

```bash
git clone https://github.com/star-ga/nnue-inference-bench.git
cd nnue-inference-bench
python -m pip install numpy

# Reference smoke test (pinned checksum for 10 positions)
python tools/smoke_cpu.py

# C++ scalar + AVX2
cmake -S native-cpp -B native-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-cpp/build --config Release

# CUDA (optional; builds for the GPU present unless CMAKE_CUDA_ARCHITECTURES is set)
cmake -S native-cuda -B native-cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-cuda/build --config Release

# MIND (optional; builds mind/build/libnnue_mind.so with the mindc on PATH or $MINDC)
python tools/infer_mind.py

# Run everything that is available, verify, and write the results JSON
python tools/run_benchmark.py -o build/work/results.json
python tools/validate_results.py build/work/results.json
```

`run_benchmark.py` reports a path as `unavailable` (with the reason) when its
toolchain or hardware is missing, and exits non-zero if any path that ran is not
bit-identical or a path listed in `--require` did not run. `make help` lists the
same steps as Make targets.

## Results

<!-- AUTO:BENCHMARK_START -->
| Implementation | Device | Best batch (ms) | p50 batch (ms) | Throughput (pos/s) | vs C++ scalar | Trace checksum | Bit-identical |
|---|---|---:|---:|---:|---:|---|---|
| Python reference | CPU | 162.880 | 727.784 | 6,140 | 0.15x | `0x0613C9FD` | yes |
| C++ scalar | CPU | 24.438 | 173.738 | 40,920 | 1.00x | `0x0613C9FD` | yes |
| C++ AVX2 | CPU | 4.510 | 10.077 | 221,722 | 5.42x | `0x0613C9FD` | yes |
| MIND | CPU | 161.107 | 1742.017 | 6,207 | 0.15x | `0x0613C9FD` | yes |
| CUDA | GPU | 0.279 | 0.477 | 3,584,088 | 87.59x | `0x0613C9FD` | yes |

Reference checksum `0x0613C9FD` (batch 1000, seed 42, 10 warmup + 50 measured iterations). Run 2026-10-10 with `python tools/run_benchmark.py --require python,cpp,simd,mind,cuda --note 'desktop x86-64 CPU with AVX2 and a consumer NVIDIA GPU; the machine was shared with other heavy jobs during the run (load average above the core count), so p50 times are inflated and the best batch is the better estimate' -o results/latest.json`.

Environment: Linux x86_64, Python 3.14.7, NumPy 2.5.3, c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0, Cuda compilation tools, release 13.4, V13.4.92, mind 0.10.2. Throughput uses the fastest measured batch.

Note: desktop x86-64 CPU with AVX2 and a consumer NVIDIA GPU; the machine was shared with other heavy jobs during the run (load average above the core count), so p50 times are inflated and the best batch is the better estimate.
<!-- AUTO:BENCHMARK_END -->

The table is generated from [`results/latest.json`](results/latest.json) by
`python tools/update_docs.py`; CI runs `python tools/update_docs.py --check` so the
table cannot drift from the committed results. Throughput is batch size divided by
the batch time, and each batch time covers what the row's `timing_scope` in the JSON
says: in-memory forward passes for the CPU paths, one call per batch for MIND, and
host-to-device input copy + kernels + device-to-host trace copy for CUDA.

## Verification

All paths follow one numerics contract: IEEE-754 binary32, every multiply and every
add rounded on its own (no fused multiply-add), dense sums accumulated in input-index
order starting from the bias. Under that contract the results are exactly
reproducible, so the check is equality, not a tolerance.

The checked trace has 68 float32 values per position: the eval output, the three WDL
logits, and the L2 and L3 pre-activations (32 each). The pre-activations are included
because, read with the layout in [`include/nknn_v2.md`](include/nknn_v2.md), the
bundled model's eval-head weights and bias are all zero and its WDL head barely
varies across random positions, so a checksum of the outputs alone would accept
almost any implementation. The checksum is the first 32 bits of the MD5 of the
little-endian trace array.

| Inputs | Reference checksum |
|---|---|
| 10 positions, seed 42 | `0xE67A3DAE` |
| 1,000 positions, seed 42 | `0x0613C9FD` |

Positions come from `numpy.random.default_rng(seed + i)`; the Python harness packs
their HalfKP features once and every path consumes the same packed inputs.

## How it uses MIND

`mind/nnue.mind` is the full forward pass written in MIND: sparse L1 accumulation,
SCReLU, both dense layers and both heads. `tools/infer_mind.py` compiles it with

```bash
mindc mind/nnue.mind --emit-shared mind/build/libnnue_mind.so
```

and calls the exported `nnue_run` through MIND's static-shape tensor C ABI, where
each `tensor<f64[N]>` parameter is passed as a memref descriptor
`(alloc_ptr, aligned_ptr, offset, size, stride)`. The kernel computes in `f64` and
rounds every multiply and add to `f32` (`((x) as f32) as f64`), which reproduces
binary32 arithmetic exactly, so its trace is bit-identical to the other paths.

One call evaluates a whole batch for timing; verification reads the trace back one
value per call, because exported functions return scalars. The kernel is a plain
scalar loop nest (no hand vectorisation), and in the run above it is about as fast as
the NumPy reference and slower than the C++ scalar path; the point of this row today
is that MIND reproduces the reference bit for bit, not that it is the fastest path. Language reference:
[mindlang.dev/docs](https://mindlang.dev/docs).

## Model

`models/nikola_d12v2_gold.nknn`, 20,989,768 bytes, SHA-256
`041BC71A527B830E54A83664BD7B2C4414A9F047348943DF2CE3359B49142E3C`.
The documented layout covers the first 20,989,712 bytes; the loaders ignore the
remaining 56. This repository measures inference cost and cross-implementation
agreement only; it makes no claim about evaluation quality.

## Repository layout

```
tools/            Python reference, harness, MIND and CUDA drivers, validators
native-cpp/       C++ scalar and AVX2 benchmarks (CMake)
native-cuda/      CUDA kernels (CMake, loaded with ctypes)
mind/             MIND kernel
models/           NKNN v2 model files
include/          NKNN v2 format specification
results/          Committed results behind the README table
docs/             Architecture and methodology notes
```

## Community

- **Discussions:** <https://github.com/star-ga/mind/discussions> — questions,
  ideas and help for nnue-inference-bench and the wider MIND ecosystem.
- **Discord:** <https://mindlang.dev/discord> — the MindLang community server.

## License

MIT, see [LICENSE](LICENSE).

## Links

- MIND language and documentation: [mindlang.dev](https://mindlang.dev)
- MIND compiler: [github.com/star-ga/mind](https://github.com/star-ga/mind)
