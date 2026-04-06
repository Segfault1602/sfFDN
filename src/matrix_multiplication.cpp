#include "matrix_multiplication.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#ifdef __cpp_lib_mdspan
#include <mdspan>
#endif

#include <Eigen/Core>

namespace
{

void HadamardMultiply4(std::span<const float> in, std::span<float> out)
{
    assert(in.size() % 4 == 0);
    assert(out.size() % 4 == 0);
    assert(in.size() == out.size());

    const size_t block_size = in.size() / 4;

    Eigen::Map<const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> in_map(in.data(), 4, block_size);
    Eigen::Map<Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> out_map(out.data(), 4, block_size);

    out_map.row(0) = in_map.row(0) + in_map.row(1) + in_map.row(2) + in_map.row(3);
    out_map.row(1) = in_map.row(0) - in_map.row(1) + in_map.row(2) - in_map.row(3);
    out_map.row(2) = in_map.row(0) + in_map.row(1) - in_map.row(2) - in_map.row(3);
    out_map.row(3) = in_map.row(0) - in_map.row(1) - in_map.row(2) + in_map.row(3);

    out_map *= 0.5f;
}

void HadamardMultiply8(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == 8 && out.size() == 8);

    out[0] = in[0] + in[4];
    out[1] = in[1] + in[5];
    out[2] = in[2] + in[6];
    out[3] = in[3] + in[7];
    out[4] = in[0] - in[4];
    out[5] = in[1] - in[5];
    out[6] = in[2] - in[6];
    out[7] = in[3] - in[7];

    auto a = out[0] + out[2];
    auto b = out[1] + out[3];
    auto c = out[0] - out[2];
    auto d = out[1] - out[3];
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;

    a = out[4] + out[6];
    b = out[5] + out[7];
    c = out[4] - out[6];
    d = out[5] - out[7];
    out[4] = a;
    out[5] = b;
    out[6] = c;
    out[7] = d;

    a = out[0] + out[1];
    b = out[0] - out[1];
    out[0] = a;
    out[1] = b;

    a = out[2] + out[3];
    b = out[2] - out[3];
    out[2] = a;
    out[3] = b;

    a = out[4] + out[5];
    b = out[4] - out[5];
    out[4] = a;
    out[5] = b;

    a = out[6] + out[7];
    b = out[6] - out[7];
    out[6] = a;
    out[7] = b;

    constexpr float kNormalizationFactor = 0.353553390593274f; // 1.f / std::sqrt(8.f);
    for (auto i = 0u; i < 8; ++i)
    {
        out[i] *= kNormalizationFactor;
    }
}

void HadamardMultiply16(const std::span<const float> in, std::span<float> out)
{
    out[0] = in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] + in[8] + in[9] + in[10] + in[11] + in[12] +
             in[13] + in[14] + in[15];
    out[1] = in[0] - in[1] + in[2] - in[3] + in[4] - in[5] + in[6] - in[7] + in[8] - in[9] + in[10] - in[11] + in[12] -
             in[13] + in[14] - in[15];
    out[2] = in[0] + in[1] - in[2] - in[3] + in[4] + in[5] - in[6] - in[7] + in[8] + in[9] - in[10] - in[11] + in[12] +
             in[13] - in[14] - in[15];
    out[3] = in[0] - in[1] - in[2] + in[3] + in[4] - in[5] - in[6] + in[7] + in[8] - in[9] - in[10] + in[11] + in[12] -
             in[13] - in[14] + in[15];
    out[4] = in[0] + in[1] + in[2] + in[3] - in[4] - in[5] - in[6] - in[7] + in[8] + in[9] + in[10] + in[11] - in[12] -
             in[13] - in[14] - in[15];
    out[5] = in[0] - in[1] + in[2] - in[3] - in[4] + in[5] - in[6] + in[7] + in[8] - in[9] + in[10] - in[11] - in[12] +
             in[13] - in[14] + in[15];
    out[6] = in[0] + in[1] - in[2] - in[3] - in[4] - in[5] + in[6] + in[7] + in[8] + in[9] - in[10] - in[11] - in[12] -
             in[13] + in[14] + in[15];
    out[7] = in[0] - in[1] - in[2] + in[3] - in[4] + in[5] + in[6] - in[7] + in[8] - in[9] - in[10] + in[11] - in[12] +
             in[13] + in[14] - in[15];
    out[8] = in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] - in[8] - in[9] - in[10] - in[11] - in[12] -
             in[13] - in[14] - in[15];
    out[9] = in[0] - in[1] + in[2] - in[3] + in[4] - in[5] + in[6] - in[7] - in[8] + in[9] - in[10] + in[11] - in[12] +
             in[13] - in[14] + in[15];
    out[10] = in[0] + in[1] - in[2] - in[3] + in[4] + in[5] - in[6] - in[7] - in[8] - in[9] + in[10] + in[11] - in[12] -
              in[13] + in[14] + in[15];
    out[11] = in[0] - in[1] - in[2] + in[3] + in[4] - in[5] - in[6] + in[7] - in[8] + in[9] + in[10] - in[11] - in[12] +
              in[13] + in[14] - in[15];
    out[12] = in[0] + in[1] + in[2] + in[3] - in[4] - in[5] - in[6] - in[7] - in[8] - in[9] - in[10] - in[11] + in[12] +
              in[13] + in[14] + in[15];
    out[13] = in[0] - in[1] + in[2] - in[3] - in[4] + in[5] - in[6] + in[7] - in[8] + in[9] - in[10] + in[11] + in[12] -
              in[13] + in[14] - in[15];
    out[14] = in[0] + in[1] - in[2] - in[3] - in[4] - in[5] + in[6] + in[7] - in[8] - in[9] + in[10] + in[11] + in[12] +
              in[13] - in[14] - in[15];
    out[15] = in[0] - in[1] - in[2] + in[3] - in[4] + in[5] + in[6] - in[7] - in[8] + in[9] + in[10] - in[11] + in[12] -
              in[13] - in[14] + in[15];

    for (auto i = 0u; i < 16; i += 4)
    {
        out[i] *= 0.25f;
        out[i + 1] *= 0.25f;
        out[i + 2] *= 0.25f;
        out[i + 3] *= 0.25f;
    }
}

} // namespace

namespace sfFDN
{

void HadamardMultiply(const std::span<const float> input, std::span<float> output)
{
    switch (input.size())
    {
    case 4:
        HadamardMultiply4(input, output);
        break;
    case 8:
        HadamardMultiply8(input, output);
        break;
    case 16:
        HadamardMultiply16(input, output);
        break;
    default:
        assert(false && "Unsupported size for Hadamard multiplication");
        break;
    }
}

void WalshHadamardTransform(std::span<float> inout)
{
    assert(inout.size() == 4 || inout.size() == 8 || inout.size() == 16);

    for (auto h = 1u; h < inout.size(); h *= 2)
    {
        for (auto i = 0u; i < inout.size(); i += 2 * h)
        {
            for (auto j = 0u; j < h; ++j)
            {
                const float a = inout[i + j];
                const float b = inout[i + j + h];
                inout[i + j] = a + b;
                inout[i + j + h] = a - b;
            }
        }
    }

    const float normalization_factor = 1.f / std::sqrt(static_cast<float>(inout.size()));
    for (float& i : inout)
    {
        i *= normalization_factor;
    }
}

void MatrixMultiply_16(std::span<const float, 16> in, std::span<float, 16> out,
                       const std::span<const float, 16 * 16> matrix)
{
    auto To1D = [](int x, int y) constexpr -> size_t { return (y * 16) + x; };

    out[0] = in[0] * matrix[To1D(0, 0)] + in[1] * matrix[To1D(0, 1)] + in[2] * matrix[To1D(0, 2)] +
             in[3] * matrix[To1D(0, 3)] + in[4] * matrix[To1D(0, 4)] + in[5] * matrix[To1D(0, 5)] +
             in[6] * matrix[To1D(0, 6)] + in[7] * matrix[To1D(0, 7)] + in[8] * matrix[To1D(0, 8)] +
             in[9] * matrix[To1D(0, 9)] + in[10] * matrix[To1D(0, 10)] + in[11] * matrix[To1D(0, 11)] +
             in[12] * matrix[To1D(0, 12)] + in[13] * matrix[To1D(0, 13)] + in[14] * matrix[To1D(0, 14)] +
             in[15] * matrix[To1D(0, 15)];

    out[1] = in[0] * matrix[To1D(1, 0)] + in[1] * matrix[To1D(1, 1)] + in[2] * matrix[To1D(1, 2)] +
             in[3] * matrix[To1D(1, 3)] + in[4] * matrix[To1D(1, 4)] + in[5] * matrix[To1D(1, 5)] +
             in[6] * matrix[To1D(1, 6)] + in[7] * matrix[To1D(1, 7)] + in[8] * matrix[To1D(1, 8)] +
             in[9] * matrix[To1D(1, 9)] + in[10] * matrix[To1D(1, 10)] + in[11] * matrix[To1D(1, 11)] +
             in[12] * matrix[To1D(1, 12)] + in[13] * matrix[To1D(1, 13)] + in[14] * matrix[To1D(1, 14)] +
             in[15] * matrix[To1D(1, 15)];

    out[2] = in[0] * matrix[To1D(2, 0)] + in[1] * matrix[To1D(2, 1)] + in[2] * matrix[To1D(2, 2)] +
             in[3] * matrix[To1D(2, 3)] + in[4] * matrix[To1D(2, 4)] + in[5] * matrix[To1D(2, 5)] +
             in[6] * matrix[To1D(2, 6)] + in[7] * matrix[To1D(2, 7)] + in[8] * matrix[To1D(2, 8)] +
             in[9] * matrix[To1D(2, 9)] + in[10] * matrix[To1D(2, 10)] + in[11] * matrix[To1D(2, 11)] +
             in[12] * matrix[To1D(2, 12)] + in[13] * matrix[To1D(2, 13)] + in[14] * matrix[To1D(2, 14)] +
             in[15] * matrix[To1D(2, 15)];

    out[3] = in[0] * matrix[To1D(3, 0)] + in[1] * matrix[To1D(3, 1)] + in[2] * matrix[To1D(3, 2)] +
             in[3] * matrix[To1D(3, 3)] + in[4] * matrix[To1D(3, 4)] + in[5] * matrix[To1D(3, 5)] +
             in[6] * matrix[To1D(3, 6)] + in[7] * matrix[To1D(3, 7)] + in[8] * matrix[To1D(3, 8)] +
             in[9] * matrix[To1D(3, 9)] + in[10] * matrix[To1D(3, 10)] + in[11] * matrix[To1D(3, 11)] +
             in[12] * matrix[To1D(3, 12)] + in[13] * matrix[To1D(3, 13)] + in[14] * matrix[To1D(3, 14)] +
             in[15] * matrix[To1D(3, 15)];

    out[4] = in[0] * matrix[To1D(4, 0)] + in[1] * matrix[To1D(4, 1)] + in[2] * matrix[To1D(4, 2)] +
             in[3] * matrix[To1D(4, 3)] + in[4] * matrix[To1D(4, 4)] + in[5] * matrix[To1D(4, 5)] +
             in[6] * matrix[To1D(4, 6)] + in[7] * matrix[To1D(4, 7)] + in[8] * matrix[To1D(4, 8)] +
             in[9] * matrix[To1D(4, 9)] + in[10] * matrix[To1D(4, 10)] + in[11] * matrix[To1D(4, 11)] +
             in[12] * matrix[To1D(4, 12)] + in[13] * matrix[To1D(4, 13)] + in[14] * matrix[To1D(4, 14)] +
             in[15] * matrix[To1D(4, 15)];

    out[5] = in[0] * matrix[To1D(5, 0)] + in[1] * matrix[To1D(5, 1)] + in[2] * matrix[To1D(5, 2)] +
             in[3] * matrix[To1D(5, 3)] + in[4] * matrix[To1D(5, 4)] + in[5] * matrix[To1D(5, 5)] +
             in[6] * matrix[To1D(5, 6)] + in[7] * matrix[To1D(5, 7)] + in[8] * matrix[To1D(5, 8)] +
             in[9] * matrix[To1D(5, 9)] + in[10] * matrix[To1D(5, 10)] + in[11] * matrix[To1D(5, 11)] +
             in[12] * matrix[To1D(5, 12)] + in[13] * matrix[To1D(5, 13)] + in[14] * matrix[To1D(5, 14)] +
             in[15] * matrix[To1D(5, 15)];

    out[6] = in[0] * matrix[To1D(6, 0)] + in[1] * matrix[To1D(6, 1)] + in[2] * matrix[To1D(6, 2)] +
             in[3] * matrix[To1D(6, 3)] + in[4] * matrix[To1D(6, 4)] + in[5] * matrix[To1D(6, 5)] +
             in[6] * matrix[To1D(6, 6)] + in[7] * matrix[To1D(6, 7)] + in[8] * matrix[To1D(6, 8)] +
             in[9] * matrix[To1D(6, 9)] + in[10] * matrix[To1D(6, 10)] + in[11] * matrix[To1D(6, 11)] +
             in[12] * matrix[To1D(6, 12)] + in[13] * matrix[To1D(6, 13)] + in[14] * matrix[To1D(6, 14)] +
             in[15] * matrix[To1D(6, 15)];

    out[7] = in[0] * matrix[To1D(7, 0)] + in[1] * matrix[To1D(7, 1)] + in[2] * matrix[To1D(7, 2)] +
             in[3] * matrix[To1D(7, 3)] + in[4] * matrix[To1D(7, 4)] + in[5] * matrix[To1D(7, 5)] +
             in[6] * matrix[To1D(7, 6)] + in[7] * matrix[To1D(7, 7)] + in[8] * matrix[To1D(7, 8)] +
             in[9] * matrix[To1D(7, 9)] + in[10] * matrix[To1D(7, 10)] + in[11] * matrix[To1D(7, 11)] +
             in[12] * matrix[To1D(7, 12)] + in[13] * matrix[To1D(7, 13)] + in[14] * matrix[To1D(7, 14)] +
             in[15] * matrix[To1D(7, 15)];

    out[8] = in[0] * matrix[To1D(8, 0)] + in[1] * matrix[To1D(8, 1)] + in[2] * matrix[To1D(8, 2)] +
             in[3] * matrix[To1D(8, 3)] + in[4] * matrix[To1D(8, 4)] + in[5] * matrix[To1D(8, 5)] +
             in[6] * matrix[To1D(8, 6)] + in[7] * matrix[To1D(8, 7)] + in[8] * matrix[To1D(8, 8)] +
             in[9] * matrix[To1D(8, 9)] + in[10] * matrix[To1D(8, 10)] + in[11] * matrix[To1D(8, 11)] +
             in[12] * matrix[To1D(8, 12)] + in[13] * matrix[To1D(8, 13)] + in[14] * matrix[To1D(8, 14)] +
             in[15] * matrix[To1D(8, 15)];

    out[9] = in[0] * matrix[To1D(9, 0)] + in[1] * matrix[To1D(9, 1)] + in[2] * matrix[To1D(9, 2)] +
             in[3] * matrix[To1D(9, 3)] + in[4] * matrix[To1D(9, 4)] + in[5] * matrix[To1D(9, 5)] +
             in[6] * matrix[To1D(9, 6)] + in[7] * matrix[To1D(9, 7)] + in[8] * matrix[To1D(9, 8)] +
             in[9] * matrix[To1D(9, 9)] + in[10] * matrix[To1D(9, 10)] + in[11] * matrix[To1D(9, 11)] +
             in[12] * matrix[To1D(9, 12)] + in[13] * matrix[To1D(9, 13)] + in[14] * matrix[To1D(9, 14)] +
             in[15] * matrix[To1D(9, 15)];

    out[10] = in[0] * matrix[To1D(10, 0)] + in[1] * matrix[To1D(10, 1)] + in[2] * matrix[To1D(10, 2)] +
              in[3] * matrix[To1D(10, 3)] + in[4] * matrix[To1D(10, 4)] + in[5] * matrix[To1D(10, 5)] +
              in[6] * matrix[To1D(10, 6)] + in[7] * matrix[To1D(10, 7)] + in[8] * matrix[To1D(10, 8)] +
              in[9] * matrix[To1D(10, 9)] + in[10] * matrix[To1D(10, 10)] + in[11] * matrix[To1D(10, 11)] +
              in[12] * matrix[To1D(10, 12)] + in[13] * matrix[To1D(10, 13)] + in[14] * matrix[To1D(10, 14)] +
              in[15] * matrix[To1D(10, 15)];

    out[11] = in[0] * matrix[To1D(3, 0)] + in[1] * matrix[To1D(3, 1)] + in[2] * matrix[To1D(3, 2)] +
              in[3] * matrix[To1D(3, 3)] + in[4] * matrix[To1D(3, 4)] + in[5] * matrix[To1D(3, 5)] +
              in[6] * matrix[To1D(3, 6)] + in[7] * matrix[To1D(3, 7)] + in[8] * matrix[To1D(3, 8)] +
              in[9] * matrix[To1D(3, 9)] + in[10] * matrix[To1D(3, 10)] + in[11] * matrix[To1D(3, 11)] +
              in[12] * matrix[To1D(3, 12)] + in[13] * matrix[To1D(3, 13)] + in[14] * matrix[To1D(3, 14)] +
              in[15] * matrix[To1D(3, 15)];

    out[12] = in[0] * matrix[To1D(12, 0)] + in[1] * matrix[To1D(12, 1)] + in[2] * matrix[To1D(12, 2)] +
              in[3] * matrix[To1D(12, 3)] + in[4] * matrix[To1D(12, 4)] + in[5] * matrix[To1D(12, 5)] +
              in[6] * matrix[To1D(12, 6)] + in[7] * matrix[To1D(12, 7)] + in[8] * matrix[To1D(12, 8)] +
              in[9] * matrix[To1D(12, 9)] + in[10] * matrix[To1D(12, 10)] + in[11] * matrix[To1D(12, 11)] +
              in[12] * matrix[To1D(12, 12)] + in[13] * matrix[To1D(12, 13)] + in[14] * matrix[To1D(12, 14)] +
              in[15] * matrix[To1D(12, 15)];

    out[13] = in[0] * matrix[To1D(13, 0)] + in[1] * matrix[To1D(13, 1)] + in[2] * matrix[To1D(13, 2)] +
              in[3] * matrix[To1D(13, 3)] + in[4] * matrix[To1D(13, 4)] + in[5] * matrix[To1D(13, 5)] +
              in[6] * matrix[To1D(13, 6)] + in[7] * matrix[To1D(13, 7)] + in[8] * matrix[To1D(13, 8)] +
              in[9] * matrix[To1D(13, 9)] + in[10] * matrix[To1D(13, 10)] + in[11] * matrix[To1D(13, 11)] +
              in[12] * matrix[To1D(13, 12)] + in[13] * matrix[To1D(13, 13)] + in[14] * matrix[To1D(13, 14)] +
              in[15] * matrix[To1D(13, 15)];

    out[14] = in[0] * matrix[To1D(14, 0)] + in[1] * matrix[To1D(14, 1)] + in[2] * matrix[To1D(14, 2)] +
              in[3] * matrix[To1D(14, 3)] + in[4] * matrix[To1D(14, 4)] + in[5] * matrix[To1D(14, 5)] +
              in[6] * matrix[To1D(14, 6)] + in[7] * matrix[To1D(14, 7)] + in[8] * matrix[To1D(14, 8)] +
              in[9] * matrix[To1D(14, 9)] + in[10] * matrix[To1D(14, 10)] + in[11] * matrix[To1D(14, 11)] +
              in[12] * matrix[To1D(14, 12)] + in[13] * matrix[To1D(14, 13)] + in[14] * matrix[To1D(14, 14)] +
              in[15] * matrix[To1D(14, 15)];

    out[15] = in[0] * matrix[To1D(15, 0)] + in[1] * matrix[To1D(15, 1)] + in[2] * matrix[To1D(15, 2)] +
              in[3] * matrix[To1D(15, 3)] + in[4] * matrix[To1D(15, 4)] + in[5] * matrix[To1D(15, 5)] +
              in[6] * matrix[To1D(15, 6)] + in[7] * matrix[To1D(15, 7)] + in[8] * matrix[To1D(15, 8)] +
              in[9] * matrix[To1D(15, 9)] + in[10] * matrix[To1D(15, 10)] + in[11] * matrix[To1D(15, 11)] +
              in[12] * matrix[To1D(15, 12)] + in[13] * matrix[To1D(15, 13)] + in[14] * matrix[To1D(15, 14)] +
              in[15] * matrix[To1D(15, 15)];
}

void MatrixMultiply_C(std::span<const float> in, std::span<float> out, std::span<const float> matrix, uint32_t mat_size)
{
    // Everything is in col-major order.

    const uint32_t row_count = in.size() / mat_size;
    const uint32_t col_count = mat_size;

    for (auto k = 0u; k < row_count; ++k)
    {
        const uint32_t offset = k;
        for (auto i = 0u; i < mat_size; ++i)
        {
            out[(i * row_count) + offset] = 0.0f;

            const uint32_t unroll_size = mat_size & ~7;
            uint32_t idx = 0;
            for (; idx < unroll_size; idx += 8)
            {
                const auto in_offset = k + (idx * row_count);
                const auto mat_offset = (i * col_count) + idx;
                const auto out_idx = (i * row_count) + offset;

                out[out_idx] += in[in_offset] * matrix[mat_offset] +
                                in[in_offset + (1 * row_count)] * matrix[mat_offset + 1] +
                                in[in_offset + (2 * row_count)] * matrix[mat_offset + 2] +
                                in[in_offset + (3 * row_count)] * matrix[mat_offset + 3] +
                                in[in_offset + (4 * row_count)] * matrix[mat_offset + 4] +
                                in[in_offset + (5 * row_count)] * matrix[mat_offset + 5] +
                                in[in_offset + (6 * row_count)] * matrix[mat_offset + 6] +
                                in[in_offset + (7 * row_count)] * matrix[mat_offset + 7];
            }

            for (; idx < mat_size; ++idx)
            {
                out[(i * row_count) + offset] += in[k + (idx * row_count)] * matrix[(i * col_count) + idx];
            }
        }
    }
}

} // namespace sfFDN