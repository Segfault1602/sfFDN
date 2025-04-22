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
    auto to_1d = [](int x, int y) constexpr -> size_t { return (y * 16) + x; };

    out[0] = in[0] * matrix[to_1d(0, 0)] + in[1] * matrix[to_1d(0, 1)] + in[2] * matrix[to_1d(0, 2)] +
             in[3] * matrix[to_1d(0, 3)] + in[4] * matrix[to_1d(0, 4)] + in[5] * matrix[to_1d(0, 5)] +
             in[6] * matrix[to_1d(0, 6)] + in[7] * matrix[to_1d(0, 7)] + in[8] * matrix[to_1d(0, 8)] +
             in[9] * matrix[to_1d(0, 9)] + in[10] * matrix[to_1d(0, 10)] + in[11] * matrix[to_1d(0, 11)] +
             in[12] * matrix[to_1d(0, 12)] + in[13] * matrix[to_1d(0, 13)] + in[14] * matrix[to_1d(0, 14)] +
             in[15] * matrix[to_1d(0, 15)];

    out[1] = in[0] * matrix[to_1d(1, 0)] + in[1] * matrix[to_1d(1, 1)] + in[2] * matrix[to_1d(1, 2)] +
             in[3] * matrix[to_1d(1, 3)] + in[4] * matrix[to_1d(1, 4)] + in[5] * matrix[to_1d(1, 5)] +
             in[6] * matrix[to_1d(1, 6)] + in[7] * matrix[to_1d(1, 7)] + in[8] * matrix[to_1d(1, 8)] +
             in[9] * matrix[to_1d(1, 9)] + in[10] * matrix[to_1d(1, 10)] + in[11] * matrix[to_1d(1, 11)] +
             in[12] * matrix[to_1d(1, 12)] + in[13] * matrix[to_1d(1, 13)] + in[14] * matrix[to_1d(1, 14)] +
             in[15] * matrix[to_1d(1, 15)];

    out[2] = in[0] * matrix[to_1d(2, 0)] + in[1] * matrix[to_1d(2, 1)] + in[2] * matrix[to_1d(2, 2)] +
             in[3] * matrix[to_1d(2, 3)] + in[4] * matrix[to_1d(2, 4)] + in[5] * matrix[to_1d(2, 5)] +
             in[6] * matrix[to_1d(2, 6)] + in[7] * matrix[to_1d(2, 7)] + in[8] * matrix[to_1d(2, 8)] +
             in[9] * matrix[to_1d(2, 9)] + in[10] * matrix[to_1d(2, 10)] + in[11] * matrix[to_1d(2, 11)] +
             in[12] * matrix[to_1d(2, 12)] + in[13] * matrix[to_1d(2, 13)] + in[14] * matrix[to_1d(2, 14)] +
             in[15] * matrix[to_1d(2, 15)];

    out[3] = in[0] * matrix[to_1d(3, 0)] + in[1] * matrix[to_1d(3, 1)] + in[2] * matrix[to_1d(3, 2)] +
             in[3] * matrix[to_1d(3, 3)] + in[4] * matrix[to_1d(3, 4)] + in[5] * matrix[to_1d(3, 5)] +
             in[6] * matrix[to_1d(3, 6)] + in[7] * matrix[to_1d(3, 7)] + in[8] * matrix[to_1d(3, 8)] +
             in[9] * matrix[to_1d(3, 9)] + in[10] * matrix[to_1d(3, 10)] + in[11] * matrix[to_1d(3, 11)] +
             in[12] * matrix[to_1d(3, 12)] + in[13] * matrix[to_1d(3, 13)] + in[14] * matrix[to_1d(3, 14)] +
             in[15] * matrix[to_1d(3, 15)];

    out[4] = in[0] * matrix[to_1d(4, 0)] + in[1] * matrix[to_1d(4, 1)] + in[2] * matrix[to_1d(4, 2)] +
             in[3] * matrix[to_1d(4, 3)] + in[4] * matrix[to_1d(4, 4)] + in[5] * matrix[to_1d(4, 5)] +
             in[6] * matrix[to_1d(4, 6)] + in[7] * matrix[to_1d(4, 7)] + in[8] * matrix[to_1d(4, 8)] +
             in[9] * matrix[to_1d(4, 9)] + in[10] * matrix[to_1d(4, 10)] + in[11] * matrix[to_1d(4, 11)] +
             in[12] * matrix[to_1d(4, 12)] + in[13] * matrix[to_1d(4, 13)] + in[14] * matrix[to_1d(4, 14)] +
             in[15] * matrix[to_1d(4, 15)];

    out[5] = in[0] * matrix[to_1d(5, 0)] + in[1] * matrix[to_1d(5, 1)] + in[2] * matrix[to_1d(5, 2)] +
             in[3] * matrix[to_1d(5, 3)] + in[4] * matrix[to_1d(5, 4)] + in[5] * matrix[to_1d(5, 5)] +
             in[6] * matrix[to_1d(5, 6)] + in[7] * matrix[to_1d(5, 7)] + in[8] * matrix[to_1d(5, 8)] +
             in[9] * matrix[to_1d(5, 9)] + in[10] * matrix[to_1d(5, 10)] + in[11] * matrix[to_1d(5, 11)] +
             in[12] * matrix[to_1d(5, 12)] + in[13] * matrix[to_1d(5, 13)] + in[14] * matrix[to_1d(5, 14)] +
             in[15] * matrix[to_1d(5, 15)];

    out[6] = in[0] * matrix[to_1d(6, 0)] + in[1] * matrix[to_1d(6, 1)] + in[2] * matrix[to_1d(6, 2)] +
             in[3] * matrix[to_1d(6, 3)] + in[4] * matrix[to_1d(6, 4)] + in[5] * matrix[to_1d(6, 5)] +
             in[6] * matrix[to_1d(6, 6)] + in[7] * matrix[to_1d(6, 7)] + in[8] * matrix[to_1d(6, 8)] +
             in[9] * matrix[to_1d(6, 9)] + in[10] * matrix[to_1d(6, 10)] + in[11] * matrix[to_1d(6, 11)] +
             in[12] * matrix[to_1d(6, 12)] + in[13] * matrix[to_1d(6, 13)] + in[14] * matrix[to_1d(6, 14)] +
             in[15] * matrix[to_1d(6, 15)];

    out[7] = in[0] * matrix[to_1d(7, 0)] + in[1] * matrix[to_1d(7, 1)] + in[2] * matrix[to_1d(7, 2)] +
             in[3] * matrix[to_1d(7, 3)] + in[4] * matrix[to_1d(7, 4)] + in[5] * matrix[to_1d(7, 5)] +
             in[6] * matrix[to_1d(7, 6)] + in[7] * matrix[to_1d(7, 7)] + in[8] * matrix[to_1d(7, 8)] +
             in[9] * matrix[to_1d(7, 9)] + in[10] * matrix[to_1d(7, 10)] + in[11] * matrix[to_1d(7, 11)] +
             in[12] * matrix[to_1d(7, 12)] + in[13] * matrix[to_1d(7, 13)] + in[14] * matrix[to_1d(7, 14)] +
             in[15] * matrix[to_1d(7, 15)];

    out[8] = in[0] * matrix[to_1d(8, 0)] + in[1] * matrix[to_1d(8, 1)] + in[2] * matrix[to_1d(8, 2)] +
             in[3] * matrix[to_1d(8, 3)] + in[4] * matrix[to_1d(8, 4)] + in[5] * matrix[to_1d(8, 5)] +
             in[6] * matrix[to_1d(8, 6)] + in[7] * matrix[to_1d(8, 7)] + in[8] * matrix[to_1d(8, 8)] +
             in[9] * matrix[to_1d(8, 9)] + in[10] * matrix[to_1d(8, 10)] + in[11] * matrix[to_1d(8, 11)] +
             in[12] * matrix[to_1d(8, 12)] + in[13] * matrix[to_1d(8, 13)] + in[14] * matrix[to_1d(8, 14)] +
             in[15] * matrix[to_1d(8, 15)];

    out[9] = in[0] * matrix[to_1d(9, 0)] + in[1] * matrix[to_1d(9, 1)] + in[2] * matrix[to_1d(9, 2)] +
             in[3] * matrix[to_1d(9, 3)] + in[4] * matrix[to_1d(9, 4)] + in[5] * matrix[to_1d(9, 5)] +
             in[6] * matrix[to_1d(9, 6)] + in[7] * matrix[to_1d(9, 7)] + in[8] * matrix[to_1d(9, 8)] +
             in[9] * matrix[to_1d(9, 9)] + in[10] * matrix[to_1d(9, 10)] + in[11] * matrix[to_1d(9, 11)] +
             in[12] * matrix[to_1d(9, 12)] + in[13] * matrix[to_1d(9, 13)] + in[14] * matrix[to_1d(9, 14)] +
             in[15] * matrix[to_1d(9, 15)];

    out[10] = in[0] * matrix[to_1d(10, 0)] + in[1] * matrix[to_1d(10, 1)] + in[2] * matrix[to_1d(10, 2)] +
              in[3] * matrix[to_1d(10, 3)] + in[4] * matrix[to_1d(10, 4)] + in[5] * matrix[to_1d(10, 5)] +
              in[6] * matrix[to_1d(10, 6)] + in[7] * matrix[to_1d(10, 7)] + in[8] * matrix[to_1d(10, 8)] +
              in[9] * matrix[to_1d(10, 9)] + in[10] * matrix[to_1d(10, 10)] + in[11] * matrix[to_1d(10, 11)] +
              in[12] * matrix[to_1d(10, 12)] + in[13] * matrix[to_1d(10, 13)] + in[14] * matrix[to_1d(10, 14)] +
              in[15] * matrix[to_1d(10, 15)];

    out[11] = in[0] * matrix[to_1d(3, 0)] + in[1] * matrix[to_1d(3, 1)] + in[2] * matrix[to_1d(3, 2)] +
              in[3] * matrix[to_1d(3, 3)] + in[4] * matrix[to_1d(3, 4)] + in[5] * matrix[to_1d(3, 5)] +
              in[6] * matrix[to_1d(3, 6)] + in[7] * matrix[to_1d(3, 7)] + in[8] * matrix[to_1d(3, 8)] +
              in[9] * matrix[to_1d(3, 9)] + in[10] * matrix[to_1d(3, 10)] + in[11] * matrix[to_1d(3, 11)] +
              in[12] * matrix[to_1d(3, 12)] + in[13] * matrix[to_1d(3, 13)] + in[14] * matrix[to_1d(3, 14)] +
              in[15] * matrix[to_1d(3, 15)];

    out[12] = in[0] * matrix[to_1d(12, 0)] + in[1] * matrix[to_1d(12, 1)] + in[2] * matrix[to_1d(12, 2)] +
              in[3] * matrix[to_1d(12, 3)] + in[4] * matrix[to_1d(12, 4)] + in[5] * matrix[to_1d(12, 5)] +
              in[6] * matrix[to_1d(12, 6)] + in[7] * matrix[to_1d(12, 7)] + in[8] * matrix[to_1d(12, 8)] +
              in[9] * matrix[to_1d(12, 9)] + in[10] * matrix[to_1d(12, 10)] + in[11] * matrix[to_1d(12, 11)] +
              in[12] * matrix[to_1d(12, 12)] + in[13] * matrix[to_1d(12, 13)] + in[14] * matrix[to_1d(12, 14)] +
              in[15] * matrix[to_1d(12, 15)];

    out[13] = in[0] * matrix[to_1d(13, 0)] + in[1] * matrix[to_1d(13, 1)] + in[2] * matrix[to_1d(13, 2)] +
              in[3] * matrix[to_1d(13, 3)] + in[4] * matrix[to_1d(13, 4)] + in[5] * matrix[to_1d(13, 5)] +
              in[6] * matrix[to_1d(13, 6)] + in[7] * matrix[to_1d(13, 7)] + in[8] * matrix[to_1d(13, 8)] +
              in[9] * matrix[to_1d(13, 9)] + in[10] * matrix[to_1d(13, 10)] + in[11] * matrix[to_1d(13, 11)] +
              in[12] * matrix[to_1d(13, 12)] + in[13] * matrix[to_1d(13, 13)] + in[14] * matrix[to_1d(13, 14)] +
              in[15] * matrix[to_1d(13, 15)];

    out[14] = in[0] * matrix[to_1d(14, 0)] + in[1] * matrix[to_1d(14, 1)] + in[2] * matrix[to_1d(14, 2)] +
              in[3] * matrix[to_1d(14, 3)] + in[4] * matrix[to_1d(14, 4)] + in[5] * matrix[to_1d(14, 5)] +
              in[6] * matrix[to_1d(14, 6)] + in[7] * matrix[to_1d(14, 7)] + in[8] * matrix[to_1d(14, 8)] +
              in[9] * matrix[to_1d(14, 9)] + in[10] * matrix[to_1d(14, 10)] + in[11] * matrix[to_1d(14, 11)] +
              in[12] * matrix[to_1d(14, 12)] + in[13] * matrix[to_1d(14, 13)] + in[14] * matrix[to_1d(14, 14)] +
              in[15] * matrix[to_1d(14, 15)];

    out[15] = in[0] * matrix[to_1d(15, 0)] + in[1] * matrix[to_1d(15, 1)] + in[2] * matrix[to_1d(15, 2)] +
              in[3] * matrix[to_1d(15, 3)] + in[4] * matrix[to_1d(15, 4)] + in[5] * matrix[to_1d(15, 5)] +
              in[6] * matrix[to_1d(15, 6)] + in[7] * matrix[to_1d(15, 7)] + in[8] * matrix[to_1d(15, 8)] +
              in[9] * matrix[to_1d(15, 9)] + in[10] * matrix[to_1d(15, 10)] + in[11] * matrix[to_1d(15, 11)] +
              in[12] * matrix[to_1d(15, 12)] + in[13] * matrix[to_1d(15, 13)] + in[14] * matrix[to_1d(15, 14)] +
              in[15] * matrix[to_1d(15, 15)];
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