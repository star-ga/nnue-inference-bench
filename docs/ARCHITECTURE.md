# Architecture

## Data flow

```
seeded positions -> HalfKP features -> packed int32 rows ──┬─> Python reference (NumPy)
   (tools/infer_cpu.py)                                     ├─> C++ scalar / AVX2   (file)
                                                            ├─> MIND kernel         (ctypes)
                                                            └─> CUDA kernels        (ctypes)
                                            each path -> float32 trace [N, 68] -> compare
```

`tools/run_benchmark.py` drives every path, compares each trace with the reference,
and writes one results JSON. `tools/update_docs.py` renders the README table from
`results/latest.json`.

## Network

```
HalfKP features (40960 per perspective)
  L1  sparse 40960 -> 256 per perspective   W1 i16, B1 i16, scale 128
      SCReLU, concatenate [side to move, other side] -> 512
  L2  dense 512 -> 32, SCReLU               W2 i8 (scale 64), B2 i16 (scale 128)
  L3  dense 32 -> 32, SCReLU                W3 i8, B3 i16
  heads  eval 32 -> 1, WDL 32 -> 3          W4/W_wdl i8, B4/B_wdl i16
```

SCReLU(x) = clamp(x, 0, 1)^2. The binary layout is in [`include/nknn_v2.md`](../include/nknn_v2.md).

## Components

| File | Role |
|---|---|
| `tools/inspect_nknn.py` | NKNN v2 loader and inspector (dequantised weights, SHA-256) |
| `tools/infer_cpu.py` | reference forward pass, position generator, input packing, checksum |
| `tools/infer_mind.py` | builds `mind/nnue.mind` with `mindc --emit-shared`, calls it via ctypes |
| `tools/infer_gpu.py` | loads `native-cuda` library via ctypes; weights resident on the device |
| `tools/run_benchmark.py` | runs, verifies and times every path; writes the results JSON |
| `tools/validate_results.py` | CI gate: pinned reference checksum, every path bit-identical |
| `tools/update_docs.py` | README results table from a results JSON (`--check` for CI) |
| `native-cpp/` | `bench_cpu` (scalar) and `bench_simd` (AVX2) from one driver |
| `native-cuda/` | L1 and dense kernels plus a C ABI for ctypes |
| `mind/nnue.mind` | the full forward pass in MIND |
