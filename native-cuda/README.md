# native-cuda

CUDA kernels for the NNUE forward pass, built as a shared library that
`tools/infer_gpu.py` loads with ctypes.

```bash
cmake -S native-cuda -B native-cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-cuda/build --config Release
python tools/infer_gpu.py   # prints availability or the reason it is unavailable
```

By default CMake builds for the GPU present at configure time
(`CMAKE_CUDA_ARCHITECTURES=native`); pass e.g. `-DCMAKE_CUDA_ARCHITECTURES="75;86;89"`
for a portable build. Use a CUDA toolkit your driver supports and a host compiler
your toolkit supports (set `-DCMAKE_CUDA_COMPILER` / `-DCMAKE_CUDA_HOST_COMPILER`
if the defaults do not match).

Kernels: `kernel_l1` (sparse accumulation for both perspectives, SCReLU,
side-to-move concatenation) and `kernel_dense` (one thread per output neuron,
writes the pre-activation and optionally SCReLU). Arithmetic uses `__fmul_rn` and
`__fadd_rn` so results are bit-identical to the CPU reference.
