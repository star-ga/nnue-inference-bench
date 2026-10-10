#include "nnue_cpu.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace {

template <typename T>
bool read_values(std::ifstream& file, std::vector<T>& out, size_t count) {
    out.resize(count);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(file);
}

template <typename Q, size_t N>
bool read_dequant(std::ifstream& file, std::array<float, N>& out, size_t count, float scale,
                  size_t stride = 1, size_t offset = 0) {
    std::vector<Q> raw;
    if (!read_values(file, raw, count)) return false;
    for (size_t i = 0; i < count; i++) out[i * stride + offset] = static_cast<float>(raw[i]) / scale;
    return true;
}

inline float screlu(float x) {
    const float clipped = std::max(0.0f, std::min(1.0f, x));
    return clipped * clipped;
}

}  // namespace

bool load_nknn(const std::string& path, NKNNModel& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    uint32_t header[2] = {0, 0};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || header[0] != 0x4E4B4E4E || header[1] != 2) return false;

    std::vector<int16_t> w1;
    if (!read_values(file, w1, static_cast<size_t>(L1_INPUT) * L1_OUTPUT)) return false;
    model.W1.resize(w1.size());
    for (size_t i = 0; i < w1.size(); i++) model.W1[i] = static_cast<float>(w1[i]) / SCALE_SPARSE;

    return read_dequant<int16_t>(file, model.B1, L1_OUTPUT, SCALE_SPARSE) &&
           read_dequant<int8_t>(file, model.W2, L2_INPUT * L2_OUTPUT, SCALE_DENSE_W) &&
           read_dequant<int16_t>(file, model.B2, L2_OUTPUT, SCALE_DENSE_B) &&
           read_dequant<int8_t>(file, model.W3, L2_OUTPUT * L3_OUTPUT, SCALE_DENSE_W) &&
           read_dequant<int16_t>(file, model.B3, L3_OUTPUT, SCALE_DENSE_B) &&
           // eval head: W4 [32] -> WH[i * 4 + 0], B4 -> BH[0]
           read_dequant<int8_t>(file, model.WH, L3_OUTPUT, SCALE_DENSE_W, HEADS, 0) &&
           read_dequant<int16_t>(file, model.BH, 1, SCALE_DENSE_B) &&
           // WDL head: W_wdl [32 * 3] -> WH[i * 4 + 1 + k], B_wdl -> BH[1 + k]
           [&] {
               std::vector<int8_t> wdl;
               if (!read_values(file, wdl, L3_OUTPUT * 3)) return false;
               for (int i = 0; i < L3_OUTPUT; i++)
                   for (int k = 0; k < 3; k++)
                       model.WH[i * HEADS + 1 + k] = static_cast<float>(wdl[i * 3 + k]) / SCALE_DENSE_W;
               std::vector<int16_t> bwdl;
               if (!read_values(file, bwdl, 3)) return false;
               for (int k = 0; k < 3; k++) model.BH[1 + k] = static_cast<float>(bwdl[k]) / SCALE_DENSE_B;
               return true;
           }();
}

bool load_inputs(const std::string& path, std::vector<int32_t>& rows, int& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    char magic[4];
    uint32_t header[3] = {0, 0, 0};  // version, count, width
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || std::memcmp(magic, "NKIN", 4) != 0 || header[0] != 1 || header[2] != INPUT_WIDTH ||
        header[1] == 0 || header[1] > (1u << 20)) {
        return false;
    }
    count = static_cast<int>(header[1]);
    if (!read_values(file, rows, static_cast<size_t>(count) * INPUT_WIDTH)) return false;
    for (int p = 0; p < count; p++) {
        if (!valid_row(&rows[static_cast<size_t>(p) * INPUT_WIDTH])) return false;
    }
    return true;
}

bool valid_row(const int32_t* row) {
    const int32_t nw = row[1];
    const int32_t nb = row[2];
    if (row[0] < 0 || row[0] > 1 || nw < 0 || nw > MAX_FEATURES || nb < 0 || nb > MAX_FEATURES) return false;
    for (int f = 0; f < nw; f++) {
        if (row[WHITE_SLOT + f] < 0 || row[WHITE_SLOT + f] >= L1_INPUT) return false;
    }
    for (int f = 0; f < nb; f++) {
        if (row[BLACK_SLOT + f] < 0 || row[BLACK_SLOT + f] >= L1_INPUT) return false;
    }
    return true;
}

void forward_trace(const NKNNModel& m, const int32_t* row, float* trace) {
    std::array<float, L1_OUTPUT> acc_w, acc_b;
    std::copy(m.B1.begin(), m.B1.end(), acc_w.begin());
    std::copy(m.B1.begin(), m.B1.end(), acc_b.begin());

    for (int f = 0; f < row[1]; f++) {
        const float* w = &m.W1[static_cast<size_t>(row[WHITE_SLOT + f]) * L1_OUTPUT];
        for (int j = 0; j < L1_OUTPUT; j++) acc_w[j] += w[j];
    }
    for (int f = 0; f < row[2]; f++) {
        const float* w = &m.W1[static_cast<size_t>(row[BLACK_SLOT + f]) * L1_OUTPUT];
        for (int j = 0; j < L1_OUTPUT; j++) acc_b[j] += w[j];
    }

    const bool white_first = row[0] == 0;
    std::array<float, L2_INPUT> h1;
    for (int j = 0; j < L1_OUTPUT; j++) {
        h1[j] = screlu(white_first ? acc_w[j] : acc_b[j]);
        h1[L1_OUTPUT + j] = screlu(white_first ? acc_b[j] : acc_w[j]);
    }

    float* pre2 = trace + HEADS;
    float* pre3 = pre2 + L2_OUTPUT;
    std::array<float, L2_OUTPUT> h2;
    for (int j = 0; j < L2_OUTPUT; j++) {
        float s = m.B2[j];
        for (int i = 0; i < L2_INPUT; i++) s += h1[i] * m.W2[i * L2_OUTPUT + j];
        pre2[j] = s;
        h2[j] = screlu(s);
    }

    std::array<float, L3_OUTPUT> h3;
    for (int j = 0; j < L3_OUTPUT; j++) {
        float s = m.B3[j];
        for (int i = 0; i < L2_OUTPUT; i++) s += h2[i] * m.W3[i * L3_OUTPUT + j];
        pre3[j] = s;
        h3[j] = screlu(s);
    }

    for (int k = 0; k < HEADS; k++) {
        float s = m.BH[k];
        for (int i = 0; i < L3_OUTPUT; i++) s += h3[i] * m.WH[i * HEADS + k];
        trace[k] = s;
    }
}
