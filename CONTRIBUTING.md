# Contributing to nnue-inference-bench

## Setup

```bash
python -m pip install numpy
python tools/smoke_cpu.py
cmake -S native-cpp -B native-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-cpp/build --config Release
```

## Checks before a pull request

```bash
python tools/run_benchmark.py --impl python,cpp,simd --require python,cpp,simd -o build/work/results.json
python tools/validate_results.py build/work/results.json
python tools/update_docs.py --check
```

If you have the toolchains, also run the MIND and CUDA paths
(`python tools/run_benchmark.py` runs everything available).

## Rules

- Every implementation follows the numerics contract in
  [docs/BENCHMARK.md](docs/BENCHMARK.md) and must stay bit-identical to the reference.
- If you deliberately change the reference (`tools/infer_cpu.py`), update the pinned
  checksums in `tools/smoke_cpu.py` and `tools/validate_results.py` in the same change
  and say why.
- README numbers come only from `results/latest.json` via `python tools/update_docs.py`.
  To publish new numbers, run `tools/run_benchmark.py -o results/latest.json` with a
  `--note` describing the hardware class, then regenerate the table.

## Adding an implementation

1. Consume the packed inputs (`tools/infer_cpu.py:pack_inputs`) and produce the
   float32 `[N, 68]` trace.
2. Add a runner to `RUNNERS` and `NAMES` in `tools/run_benchmark.py`.
3. Document how it meets the numerics contract in `docs/BENCHMARK.md`.

## Code style

Python: PEP 8 with type hints. C++: C++17, no compiler-specific extensions outside
the AVX2 intrinsics file.
