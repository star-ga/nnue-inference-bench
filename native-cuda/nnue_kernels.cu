/**
 * NNUE inference CUDA kernels (NKNN v2: 40960 -> 256x2 -> 32 -> 32 -> 1 + WDL 3).
 *
 * Numerics contract shared with the CPU and MIND paths: binary32, every
 * multiply and add rounded on its own (explicit __fmul_rn / __fadd_rn, which
 * the compiler never contracts into FMA), dense sums accumulated in input
 * index order starting from the bias. The per-position trace is therefore
 * bit-identical to the CPU reference.
 *
 * Inputs use the packed row layout (int32 x 64 per position):
 *   [stm, n_white, n_black, 30 white feature slots, 30 black feature slots, pad]
 */

#include <cuda_runtime.h>
#include <stdint.h>

#if defined(_WIN32)
#define NNUE_EXPORT extern "C" __declspec(dllexport)
#else
#define NNUE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define QUANT_SCALE_W1 128.0f
#define QUANT_SCALE_DENSE 64.0f
#define QUANT_SCALE_BIAS 128.0f

#define INPUT_SIZE 40960
#define L1_SIZE 256
#define INPUT_WIDTH 64
#define MAX_FEATURES 30
#define WHITE_SLOT 3
#define BLACK_SLOT (WHITE_SLOT + MAX_FEATURES)

__device__ __forceinline__ float screlu(float x) {
    const float clamped = fminf(fmaxf(x, 0.0f), 1.0f);
    return __fmul_rn(clamped, clamped);
}

/**
 * L1: sparse accumulation for both perspectives, SCReLU, and concatenation
 * with the side to move first. One block per position, one thread per neuron.
 * Output: hidden1 [batch, 512].
 */
__global__ void kernel_l1(const int16_t* __restrict__ W1, const int16_t* __restrict__ B1,
                          const int32_t* __restrict__ rows, float* __restrict__ hidden1, int batch_size) {
    const int pos = blockIdx.x;
    const int j = threadIdx.x;
    if (pos >= batch_size || j >= L1_SIZE) return;
    const int32_t* row = rows + pos * INPUT_WIDTH;

    const float bias = (float)B1[j] / QUANT_SCALE_W1;
    float acc_w = bias;
    float acc_b = bias;
    const int nw = min(max(row[1], 0), MAX_FEATURES);
    const int nb = min(max(row[2], 0), MAX_FEATURES);
    for (int f = 0; f < nw; f++) {
        const int idx = row[WHITE_SLOT + f];
        if (idx >= 0 && idx < INPUT_SIZE) acc_w = __fadd_rn(acc_w, (float)W1[idx * L1_SIZE + j] / QUANT_SCALE_W1);
    }
    for (int f = 0; f < nb; f++) {
        const int idx = row[BLACK_SLOT + f];
        if (idx >= 0 && idx < INPUT_SIZE) acc_b = __fadd_rn(acc_b, (float)W1[idx * L1_SIZE + j] / QUANT_SCALE_W1);
    }

    float* out = hidden1 + pos * (2 * L1_SIZE);
    const bool white_first = row[0] == 0;
    out[j] = screlu(white_first ? acc_w : acc_b);
    out[L1_SIZE + j] = screlu(white_first ? acc_b : acc_w);
}

/**
 * Dense layer: pre[pos, j] = bias[j] + sum_i input[pos, i] * W[i, j] (i
 * ascending). Writes the pre-activation to `pre` and, when `post` is not
 * null, SCReLU(pre) to `post`. One block per position, one thread per output.
 */
__global__ void kernel_dense(const int8_t* __restrict__ weights, const int16_t* __restrict__ bias,
                             const float* __restrict__ input, float* __restrict__ pre,
                             float* __restrict__ post, int batch_size, int in_size, int out_size) {
    const int pos = blockIdx.x;
    const int j = threadIdx.x;
    if (pos >= batch_size || j >= out_size) return;
    float acc = (float)bias[j] / QUANT_SCALE_BIAS;
    const float* x = input + pos * in_size;
    for (int i = 0; i < in_size; i++) {
        const float w = (float)weights[i * out_size + j] / QUANT_SCALE_DENSE;
        acc = __fadd_rn(acc, __fmul_rn(x[i], w));
    }
    pre[pos * out_size + j] = acc;
    if (post != nullptr) post[pos * out_size + j] = screlu(acc);
}

NNUE_EXPORT int nnue_get_version() { return 2; }
NNUE_EXPORT int nnue_init(int device_id) { return (int)cudaSetDevice(device_id); }
NNUE_EXPORT int nnue_sync() { return (int)cudaDeviceSynchronize(); }
NNUE_EXPORT int nnue_malloc(void** ptr, size_t size) { return (int)cudaMalloc(ptr, size); }
NNUE_EXPORT int nnue_free(void* ptr) { return (int)cudaFree(ptr); }
NNUE_EXPORT int nnue_memcpy_h2d(void* dst, const void* src, size_t size) {
    return (int)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
NNUE_EXPORT int nnue_memcpy_d2h(void* dst, const void* src, size_t size) {
    return (int)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

NNUE_EXPORT int nnue_l1(const int16_t* W1, const int16_t* B1, const int32_t* rows, float* hidden1,
                        int batch_size) {
    if (batch_size <= 0) return (int)cudaErrorInvalidValue;
    kernel_l1<<<batch_size, L1_SIZE>>>(W1, B1, rows, hidden1, batch_size);
    return (int)cudaGetLastError();
}

NNUE_EXPORT int nnue_dense(const int8_t* weights, const int16_t* bias, const float* input, float* pre,
                           float* post, int batch_size, int in_size, int out_size) {
    if (batch_size <= 0 || in_size <= 0 || out_size <= 0 || out_size > 1024) return (int)cudaErrorInvalidValue;
    kernel_dense<<<batch_size, out_size>>>(weights, bias, input, pre, post, batch_size, in_size, out_size);
    return (int)cudaGetLastError();
}
