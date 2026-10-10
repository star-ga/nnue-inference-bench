# Benchmark methodology

## Inputs

- Model: `models/nikola_d12v2_gold.nknn` (SHA-256 checked by `tools/smoke_cpu.py`).
- Positions: position `i` of a batch is generated from
  `numpy.random.default_rng(seed + i)` (`tools/infer_cpu.py:create_random_position`).
  Default batch 1000, seed 42.
- Features: HalfKP indices, packed once by `pack_inputs` into int32 rows of 64 slots:
  `[stm, n_white, n_black, 30 white slots, 30 black slots, pad]`. The C++ binaries
  read the same rows from a file written by `write_inputs`; the MIND and CUDA paths
  receive them through ctypes.

## Numerics contract

IEEE-754 binary32; every multiply and every add is rounded on its own (no fused
multiply-add); dense sums are accumulated in input-index order starting from the
bias. How each path honours it:

| Path | Mechanism |
|---|---|
| Python | separate NumPy float32 multiply and add per input index |
| C++ scalar | plain loops, `-ffp-contract=off` (`/fp:precise` on MSVC) |
| C++ AVX2 | vectorised across output neurons with `_mm256_mul_ps` + `_mm256_add_ps`, no FMA |
| CUDA | `__fmul_rn` / `__fadd_rn`, built with `--fmad=false` |
| MIND | f64 arithmetic rounded to f32 after every operation |

The L1 accumulation is exact in any order (all values are multiples of 1/128 and far
below the float32 integer range), so only the dense layers depend on the order.

## Verification

Each path returns a 68-value float32 trace per position: eval, three WDL logits, the
32 L2 pre-activations and the 32 L3 pre-activations. `run_benchmark.py` compares every
trace with the Python reference bit for bit and records the MD5-based checksum.
`validate_results.py` also pins the reference checksum per (batch, seed), so a change
in the reference itself (an edit, or a NumPy/platform difference) fails CI.

A path is `verified` only when its trace is bit-identical; otherwise it is a
`mismatch` and the run fails.

## Timing

- Warmup iterations are run and discarded, then each measured iteration times one
  full batch with a monotonic clock.
- Throughput = batch size / fastest measured batch. p50, p95 and mean are recorded
  too; on a shared machine the minimum is the least disturbed measurement.
- What a batch covers differs by path and is stored as `timing_scope` in the JSON:
  in-memory forward passes for Python and C++, one ctypes call per batch for MIND,
  and input upload + kernels + trace download for CUDA.
- Results record software versions only (OS, Python, NumPy, compilers, `mindc`);
  hardware is described in the free-text `note`.
