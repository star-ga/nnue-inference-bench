// NNUE CPU benchmark driver (scalar reference, or AVX2 when built with NNUE_SIMD).
//
//   bench_cpu  --model M --inputs I --out O [--warmup N] [--iters N]
//   bench_simd --model M --inputs I --out O [--warmup N] [--iters N]
//
// Inputs come from tools/infer_cpu.py:write_inputs (the same positions the
// Python reference uses). The per-position trace of the last measured
// iteration is written to O as raw little-endian float32 [count, 68]; the
// Python harness fingerprints it. Timing JSON is printed on stdout.

#include "nnue_cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

#ifdef NNUE_SIMD
constexpr const char* IMPLEMENTATION = "C++ AVX2";
inline void forward(const NKNNModel& m, const int32_t* row, float* trace) { forward_trace_avx2(m, row, trace); }
#else
constexpr const char* IMPLEMENTATION = "C++ scalar";
inline void forward(const NKNNModel& m, const int32_t* row, float* trace) { forward_trace(m, row, trace); }
#endif

struct Options {
    std::string model = "models/nikola_d12v2_gold.nknn";
    std::string inputs;
    std::string out;
    int warmup = 10;
    int iters = 50;
};

bool parse_int(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > 1000000) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_args(int argc, char* argv[], Options& opt) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (arg == "--model") opt.model = value;
        else if (arg == "--inputs") opt.inputs = value;
        else if (arg == "--out") opt.out = value;
        else if (arg == "--warmup") { if (!parse_int(value, opt.warmup)) return false; }
        else if (arg == "--iters") { if (!parse_int(value, opt.iters)) return false; }
        else return false;
    }
    return !opt.inputs.empty() && !opt.out.empty() && opt.iters > 0;
}

void run_batch(const NKNNModel& m, const std::vector<int32_t>& rows, int count, std::vector<float>& trace) {
    for (int p = 0; p < count; p++) {
        forward(m, &rows[static_cast<size_t>(p) * INPUT_WIDTH], &trace[static_cast<size_t>(p) * TRACE_WIDTH]);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        std::cerr << "usage: " << argv[0]
                  << " --model M --inputs I --out O [--warmup N] [--iters N]" << std::endl;
        return 2;
    }

    NKNNModel model;
    if (!load_nknn(opt.model, model)) {
        std::cerr << "error: cannot load model " << opt.model << std::endl;
        return 1;
    }
    std::vector<int32_t> rows;
    int count = 0;
    if (!load_inputs(opt.inputs, rows, count)) {
        std::cerr << "error: cannot load inputs " << opt.inputs << std::endl;
        return 1;
    }

    std::vector<float> trace(static_cast<size_t>(count) * TRACE_WIDTH, 0.0f);
    for (int i = 0; i < opt.warmup; i++) run_batch(model, rows, count, trace);

    std::vector<double> times_ms(static_cast<size_t>(opt.iters));
    for (int i = 0; i < opt.iters; i++) {
        const auto start = std::chrono::steady_clock::now();
        run_batch(model, rows, count, trace);
        const auto end = std::chrono::steady_clock::now();
        times_ms[static_cast<size_t>(i)] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::ofstream out(opt.out, std::ios::binary);
    out.write(reinterpret_cast<const char*>(trace.data()), static_cast<std::streamsize>(trace.size() * sizeof(float)));
    if (!out) {
        std::cerr << "error: cannot write " << opt.out << std::endl;
        return 1;
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0.0;
    for (double t : times_ms) mean += t;
    mean /= static_cast<double>(times_ms.size());

    std::printf("{\"implementation\": \"%s\", \"device\": \"CPU\", \"batch_size\": %d, "
                "\"warmup_iters\": %d, \"measured_iters\": %d, \"times_ms\": [",
                IMPLEMENTATION, count, opt.warmup, opt.iters);
    for (size_t i = 0; i < times_ms.size(); i++) std::printf("%s%.6f", i ? ", " : "", times_ms[i]);
    std::printf("], \"mean_ms\": %.6f}\n", mean);
    return 0;
}
