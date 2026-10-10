// AVX2 NNUE forward pass.
//
// Vectorised across output neurons, so every lane performs exactly the scalar
// reference's sequence of operations: separate multiply and add (no FMA),
// inputs consumed in index order. The trace is bit-identical to
// forward_trace() in nnue_cpu.cpp.

#include "nnue_cpu.h"

#include <immintrin.h>

namespace {

inline __m256 screlu8(__m256 x) {
    const __m256 clipped = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), x));
    return _mm256_mul_ps(clipped, clipped);
}

void accumulate(const NKNNModel& m, const int32_t* features, int count, float* out) {
    for (int j = 0; j < L1_OUTPUT; j += 8) _mm256_storeu_ps(out + j, _mm256_loadu_ps(&m.B1[j]));
    for (int f = 0; f < count; f++) {
        const float* w = &m.W1[static_cast<size_t>(features[f]) * L1_OUTPUT];
        for (int j = 0; j < L1_OUTPUT; j += 8) {
            _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_loadu_ps(out + j), _mm256_loadu_ps(w + j)));
        }
    }
}

// out[0..32) = bias + sum_i in[i] * w[i * 32 + j], i ascending, mul then add.
void dense32(const float* in, int in_size, const float* w, const float* bias, float* out) {
    __m256 s0 = _mm256_loadu_ps(bias);
    __m256 s1 = _mm256_loadu_ps(bias + 8);
    __m256 s2 = _mm256_loadu_ps(bias + 16);
    __m256 s3 = _mm256_loadu_ps(bias + 24);
    for (int i = 0; i < in_size; i++) {
        const __m256 x = _mm256_set1_ps(in[i]);
        const float* row = w + i * 32;
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x, _mm256_loadu_ps(row)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x, _mm256_loadu_ps(row + 8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(x, _mm256_loadu_ps(row + 16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(x, _mm256_loadu_ps(row + 24)));
    }
    _mm256_storeu_ps(out, s0);
    _mm256_storeu_ps(out + 8, s1);
    _mm256_storeu_ps(out + 16, s2);
    _mm256_storeu_ps(out + 24, s3);
}

}  // namespace

void forward_trace_avx2(const NKNNModel& m, const int32_t* row, float* trace) {
    alignas(32) float acc_w[L1_OUTPUT];
    alignas(32) float acc_b[L1_OUTPUT];
    accumulate(m, row + WHITE_SLOT, row[1], acc_w);
    accumulate(m, row + BLACK_SLOT, row[2], acc_b);

    const float* first = row[0] == 0 ? acc_w : acc_b;
    const float* second = row[0] == 0 ? acc_b : acc_w;
    alignas(32) float h1[L2_INPUT];
    for (int j = 0; j < L1_OUTPUT; j += 8) {
        _mm256_store_ps(h1 + j, screlu8(_mm256_load_ps(first + j)));
        _mm256_store_ps(h1 + L1_OUTPUT + j, screlu8(_mm256_load_ps(second + j)));
    }

    float* pre2 = trace + HEADS;
    float* pre3 = pre2 + L2_OUTPUT;
    alignas(32) float h2[L2_OUTPUT];
    dense32(h1, L2_INPUT, m.W2.data(), m.B2.data(), pre2);
    for (int j = 0; j < L2_OUTPUT; j += 8) _mm256_store_ps(h2 + j, screlu8(_mm256_loadu_ps(pre2 + j)));

    alignas(32) float h3[L3_OUTPUT];
    dense32(h2, L2_OUTPUT, m.W3.data(), m.B3.data(), pre3);
    for (int j = 0; j < L3_OUTPUT; j += 8) _mm256_store_ps(h3 + j, screlu8(_mm256_loadu_ps(pre3 + j)));

    // Heads: 4 outputs in one SSE register, WH is [i * 4 + k].
    __m128 s = _mm_loadu_ps(m.BH.data());
    for (int i = 0; i < L3_OUTPUT; i++) {
        s = _mm_add_ps(s, _mm_mul_ps(_mm_set1_ps(h3[i]), _mm_loadu_ps(&m.WH[i * HEADS])));
    }
    _mm_storeu_ps(trace, s);
}
