cmake_minimum_required(VERSION 3.24)

# Default: build SASS for the GPU(s) present at configure time. Override with
# -DCMAKE_CUDA_ARCHITECTURES="75;86;89" for a portable build. Must be set
# before project(), which otherwise picks the compiler's default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nnue_cuda LANGUAGES CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(nnue_cuda SHARED nnue_kernels.cu)
# Loaded from Python with ctypes as nnue_cuda.so / nnue_cuda.dll.
set_target_properties(nnue_cuda PROPERTIES PREFIX "" OUTPUT_NAME nnue_cuda)
# Arithmetic uses explicit __fmul_rn/__fadd_rn; --fmad=false also keeps any
# remaining expressions uncontracted.
target_compile_options(nnue_cuda PRIVATE --fmad=false)
