cmake_minimum_required(VERSION 3.24)
project(nnue_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Strict FP: no contraction into fused multiply-add, so every path keeps the
# shared numerics contract (bit-identical trace). MSVC does not contract under
# /fp:precise.
if(MSVC)
  set(NNUE_STRICT_FP /fp:precise)
  set(NNUE_AVX2 /arch:AVX2)
else()
  set(NNUE_STRICT_FP -ffp-contract=off -fno-fast-math)
  set(NNUE_AVX2 -mavx2)
endif()

add_executable(bench_cpu bench_main.cpp nnue_cpu.cpp)
target_compile_options(bench_cpu PRIVATE ${NNUE_STRICT_FP})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(NNUE_X86_64 ON)
else()
  set(NNUE_X86_64 OFF)
endif()
option(NNUE_BUILD_AVX2 "Build the AVX2 benchmark (x86-64 only)" ${NNUE_X86_64})
if(NNUE_BUILD_AVX2)
  add_executable(bench_simd bench_main.cpp nnue_cpu.cpp nnue_cpu_simd.cpp)
  target_compile_definitions(bench_simd PRIVATE NNUE_SIMD)
  target_compile_options(bench_simd PRIVATE ${NNUE_STRICT_FP} ${NNUE_AVX2})
endif()
