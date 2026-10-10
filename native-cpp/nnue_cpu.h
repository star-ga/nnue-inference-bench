#ifndef NNUE_CPU_H
#define NNUE_CPU_H

// NKNN v2 NNUE inference, C++ reference (scalar) and AVX2 paths.
//
// Numerics contract shared with tools/infer_cpu.py, the CUDA kernels and the
// MIND kernel: IEEE-754 binary32, every multiply and every add rounded on its
// own (build with -ffp-contract=off and without FMA), dense sums accumulated
// in input-index order starting from the bias. Under that contract the
// per-position trace is bit-identical across all implementations.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

constexpr int L1_INPUT = 40960;
constexpr int L1_OUTPUT = 256;
constexpr int L2_INPUT = 512;
constexpr int L2_OUTPUT = 32;
constexpr int L3_OUTPUT = 32;
constexpr int HEADS = 4;  // eval, win, draw, loss

// Per-position verification trace: [eval, win, draw, loss,
// L2 pre-activation x32, L3 pre-activation x32].
constexpr int TRACE_WIDTH = HEADS + L2_OUTPUT + L3_OUTPUT;

// Packed input row (int32 x 64): [stm, n_white, n_black,
// 30 white feature slots, 30 black feature slots, pad].
constexpr int INPUT_WIDTH = 64;
constexpr int MAX_FEATURES = 30;
constexpr int WHITE_SLOT = 3;
constexpr int BLACK_SLOT = WHITE_SLOT + MAX_FEATURES;

constexpr float SCALE_SPARSE = 128.0f;
constexpr float SCALE_DENSE_W = 64.0f;
constexpr float SCALE_DENSE_B = 128.0f;

struct NKNNModel {
    std::vector<float> W1;                          // [40960 * 256]
    std::array<float, L1_OUTPUT> B1{};
    std::array<float, L2_INPUT * L2_OUTPUT> W2{};   // [i * 32 + j]
    std::array<float, L2_OUTPUT> B2{};
    std::array<float, L2_OUTPUT * L3_OUTPUT> W3{};  // [i * 32 + j]
    std::array<float, L3_OUTPUT> B3{};
    std::array<float, L3_OUTPUT * HEADS> WH{};      // [i * 4 + k], k: eval,w,d,l
    std::array<float, HEADS> BH{};
};

// Loads and dequantizes an NKNN v2 file. Returns false on any I/O or format error.
bool load_nknn(const std::string& path, NKNNModel& model);

// Loads a packed input file written by tools/infer_cpu.py:write_inputs.
bool load_inputs(const std::string& path, std::vector<int32_t>& rows, int& count);

// Validates one packed row (feature counts and indices in range).
bool valid_row(const int32_t* row);

// Scalar reference forward pass for one packed row; writes TRACE_WIDTH floats.
void forward_trace(const NKNNModel& model, const int32_t* row, float* trace);

#ifdef NNUE_SIMD
// AVX2 forward pass with the same operation order (bit-identical trace).
void forward_trace_avx2(const NKNNModel& model, const int32_t* row, float* trace);
#endif

#endif  // NNUE_CPU_H
