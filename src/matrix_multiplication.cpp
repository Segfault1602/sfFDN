#include "matrix_multiplication.h"

#include <cassert>
#include <iostream>
#include <mdspan>

namespace
{

void HadamardMultiply_4(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == 4 && out.size() == 4);

    out[0] = in[0] + in[1] + in[2] + in[3];
    out[1] = in[0] - in[1] + in[2] - in[3];
    out[2] = in[0] + in[1] - in[2] - in[3];
    out[3] = in[0] - in[1] - in[2] + in[3];

    out[0] *= 0.5f;
    out[1] *= 0.5f;
    out[2] *= 0.5f;
    out[3] *= 0.5f;
}

void HadamardMultiply_8(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == 8 && out.size() == 8);

    out[0] = in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];
    out[1] = in[0] - in[1] + in[2] - in[3] + in[4] - in[5] + in[6] - in[7];
    out[2] = in[0] + in[1] - in[2] - in[3] + in[4] + in[5] - in[6] - in[7];
    out[3] = in[0] - in[1] - in[2] + in[3] + in[4] - in[5] - in[6] + in[7];
    out[4] = in[0] + in[1] + in[2] + in[3] - in[4] - in[5] - in[6] - in[7];
    out[5] = in[0] - in[1] + in[2] - in[3] - in[4] + in[5] - in[6] + in[7];
    out[6] = in[0] + in[1] - in[2] - in[3] - in[4] - in[5] + in[6] + in[7];
    out[7] = in[0] - in[1] - in[2] + in[3] - in[4] + in[5] + in[6] - in[7];

    constexpr float normalizationFactor = 0.353553390593274f; // 1.f / std::sqrt(8.f);
    for (size_t i = 0; i < 8; ++i)
    {
        out[i] *= normalizationFactor;
    }
}

void HadamardMultiply_16(const std::span<const float> in, std::span<float> out)
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

    for (auto i = 0; i < 16; i += 4)
    {
        out[i] *= 0.25f;
        out[i + 1] *= 0.25f;
        out[i + 2] *= 0.25f;
        out[i + 3] *= 0.25f;
    }
}

} // namespace

namespace fdn
{

void HadamardMultiply(const std::span<const float> input, std::span<float> output)
{
    switch (input.size())
    {
    case 4:
        HadamardMultiply_4(input, output);
        break;
    case 8:
        HadamardMultiply_8(input, output);
        break;
    case 16:
        HadamardMultiply_16(input, output);
        break;
    default:
        assert(false && "Unsupported size for Hadamard multiplication");
        break;
    }
}

void WalshHadamardTransform(std::span<float> inout)
{
    assert(inout.size() == 4 || inout.size() == 8 || inout.size() == 16);
    const size_t N = inout.size();

    for (size_t h = 1; h < N; h *= 2)
    {
        for (size_t i = 0; i < N; i += 2 * h)
        {
            for (size_t j = 0; j < h; ++j)
            {
                const float a = inout[i + j];
                const float b = inout[i + j + h];
                inout[i + j] = a + b;
                inout[i + j + h] = a - b;
            }
        }
    }

    const float normalizationFactor = 1.f / std::sqrt(static_cast<float>(N));
    for (size_t i = 0; i < inout.size(); ++i)
    {
        inout[i] *= normalizationFactor;
    }
}

void MatrixMultiply_16(std::span<const float, 16> in, std::span<float, 16> out,
                       const std::span<const float, 16 * 16> matrix)
{
    auto md_matrix = std::mdspan(matrix.data(), 16, 16);

    out[0] = in[0] * md_matrix[0, 0] + in[1] * md_matrix[0, 1] + in[2] * md_matrix[0, 2] + in[3] * md_matrix[0, 3] +
             in[4] * md_matrix[0, 4] + in[5] * md_matrix[0, 5] + in[6] * md_matrix[0, 6] + in[7] * md_matrix[0, 7] +
             in[8] * md_matrix[0, 8] + in[9] * md_matrix[0, 9] + in[10] * md_matrix[0, 10] + in[11] * md_matrix[0, 11] +
             in[12] * md_matrix[0, 12] + in[13] * md_matrix[0, 13] + in[14] * md_matrix[0, 14] +
             in[15] * md_matrix[0, 15];

    out[1] = in[0] * md_matrix[1, 0] + in[1] * md_matrix[1, 1] + in[2] * md_matrix[1, 2] + in[3] * md_matrix[1, 3] +
             in[4] * md_matrix[1, 4] + in[5] * md_matrix[1, 5] + in[6] * md_matrix[1, 6] + in[7] * md_matrix[1, 7] +
             in[8] * md_matrix[1, 8] + in[9] * md_matrix[1, 9] + in[10] * md_matrix[1, 10] + in[11] * md_matrix[1, 11] +
             in[12] * md_matrix[1, 12] + in[13] * md_matrix[1, 13] + in[14] * md_matrix[1, 14] +
             in[15] * md_matrix[1, 15];

    out[2] = in[0] * md_matrix[2, 0] + in[1] * md_matrix[2, 1] + in[2] * md_matrix[2, 2] + in[3] * md_matrix[2, 3] +
             in[4] * md_matrix[2, 4] + in[5] * md_matrix[2, 5] + in[6] * md_matrix[2, 6] + in[7] * md_matrix[2, 7] +
             in[8] * md_matrix[2, 8] + in[9] * md_matrix[2, 9] + in[10] * md_matrix[2, 10] + in[11] * md_matrix[2, 11] +
             in[12] * md_matrix[2, 12] + in[13] * md_matrix[2, 13] + in[14] * md_matrix[2, 14] +
             in[15] * md_matrix[2, 15];

    out[3] = in[0] * md_matrix[3, 0] + in[1] * md_matrix[3, 1] + in[2] * md_matrix[3, 2] + in[3] * md_matrix[3, 3] +
             in[4] * md_matrix[3, 4] + in[5] * md_matrix[3, 5] + in[6] * md_matrix[3, 6] + in[7] * md_matrix[3, 7] +
             in[8] * md_matrix[3, 8] + in[9] * md_matrix[3, 9] + in[10] * md_matrix[3, 10] + in[11] * md_matrix[3, 11] +
             in[12] * md_matrix[3, 12] + in[13] * md_matrix[3, 13] + in[14] * md_matrix[3, 14] +
             in[15] * md_matrix[3, 15];

    out[4] = in[0] * md_matrix[4, 0] + in[1] * md_matrix[4, 1] + in[2] * md_matrix[4, 2] + in[3] * md_matrix[4, 3] +
             in[4] * md_matrix[4, 4] + in[5] * md_matrix[4, 5] + in[6] * md_matrix[4, 6] + in[7] * md_matrix[4, 7] +
             in[8] * md_matrix[4, 8] + in[9] * md_matrix[4, 9] + in[10] * md_matrix[4, 10] + in[11] * md_matrix[4, 11] +
             in[12] * md_matrix[4, 12] + in[13] * md_matrix[4, 13] + in[14] * md_matrix[4, 14] +
             in[15] * md_matrix[4, 15];

    out[5] = in[0] * md_matrix[5, 0] + in[1] * md_matrix[5, 1] + in[2] * md_matrix[5, 2] + in[3] * md_matrix[5, 3] +
             in[4] * md_matrix[5, 4] + in[5] * md_matrix[5, 5] + in[6] * md_matrix[5, 6] + in[7] * md_matrix[5, 7] +
             in[8] * md_matrix[5, 8] + in[9] * md_matrix[5, 9] + in[10] * md_matrix[5, 10] + in[11] * md_matrix[5, 11] +
             in[12] * md_matrix[5, 12] + in[13] * md_matrix[5, 13] + in[14] * md_matrix[5, 14] +
             in[15] * md_matrix[5, 15];

    out[6] = in[0] * md_matrix[6, 0] + in[1] * md_matrix[6, 1] + in[2] * md_matrix[6, 2] + in[3] * md_matrix[6, 3] +
             in[4] * md_matrix[6, 4] + in[5] * md_matrix[6, 5] + in[6] * md_matrix[6, 6] + in[7] * md_matrix[6, 7] +
             in[8] * md_matrix[6, 8] + in[9] * md_matrix[6, 9] + in[10] * md_matrix[6, 10] + in[11] * md_matrix[6, 11] +
             in[12] * md_matrix[6, 12] + in[13] * md_matrix[6, 13] + in[14] * md_matrix[6, 14] +
             in[15] * md_matrix[6, 15];

    out[7] = in[0] * md_matrix[7, 0] + in[1] * md_matrix[7, 1] + in[2] * md_matrix[7, 2] + in[3] * md_matrix[7, 3] +
             in[4] * md_matrix[7, 4] + in[5] * md_matrix[7, 5] + in[6] * md_matrix[7, 6] + in[7] * md_matrix[7, 7] +
             in[8] * md_matrix[7, 8] + in[9] * md_matrix[7, 9] + in[10] * md_matrix[7, 10] + in[11] * md_matrix[7, 11] +
             in[12] * md_matrix[7, 12] + in[13] * md_matrix[7, 13] + in[14] * md_matrix[7, 14] +
             in[15] * md_matrix[7, 15];

    out[8] = in[0] * md_matrix[8, 0] + in[1] * md_matrix[8, 1] + in[2] * md_matrix[8, 2] + in[3] * md_matrix[8, 3] +
             in[4] * md_matrix[8, 4] + in[5] * md_matrix[8, 5] + in[6] * md_matrix[8, 6] + in[7] * md_matrix[8, 7] +
             in[8] * md_matrix[8, 8] + in[9] * md_matrix[8, 9] + in[10] * md_matrix[8, 10] + in[11] * md_matrix[8, 11] +
             in[12] * md_matrix[8, 12] + in[13] * md_matrix[8, 13] + in[14] * md_matrix[8, 14] +
             in[15] * md_matrix[8, 15];

    out[9] = in[0] * md_matrix[9, 0] + in[1] * md_matrix[9, 1] + in[2] * md_matrix[9, 2] + in[3] * md_matrix[9, 3] +
             in[4] * md_matrix[9, 4] + in[5] * md_matrix[9, 5] + in[6] * md_matrix[9, 6] + in[7] * md_matrix[9, 7] +
             in[8] * md_matrix[9, 8] + in[9] * md_matrix[9, 9] + in[10] * md_matrix[9, 10] + in[11] * md_matrix[9, 11] +
             in[12] * md_matrix[9, 12] + in[13] * md_matrix[9, 13] + in[14] * md_matrix[9, 14] +
             in[15] * md_matrix[9, 15];

    out[10] = in[0] * md_matrix[10, 0] + in[1] * md_matrix[10, 1] + in[2] * md_matrix[10, 2] +
              in[3] * md_matrix[10, 3] + in[4] * md_matrix[10, 4] + in[5] * md_matrix[10, 5] +
              in[6] * md_matrix[10, 6] + in[7] * md_matrix[10, 7] + in[8] * md_matrix[10, 8] +
              in[9] * md_matrix[10, 9] + in[10] * md_matrix[10, 10] + in[11] * md_matrix[10, 11] +
              in[12] * md_matrix[10, 12] + in[13] * md_matrix[10, 13] + in[14] * md_matrix[10, 14] +
              in[15] * md_matrix[10, 15];

    out[11] = in[0] * md_matrix[11, 0] + in[1] * md_matrix[11, 1] + in[2] * md_matrix[11, 2] +
              in[3] * md_matrix[11, 3] + in[4] * md_matrix[11, 4] + in[5] * md_matrix[11, 5] +
              in[6] * md_matrix[11, 6] + in[7] * md_matrix[11, 7] + in[8] * md_matrix[11, 8] +
              in[9] * md_matrix[11, 9] + in[10] * md_matrix[11, 10] + in[11] * md_matrix[11, 11] +
              in[12] * md_matrix[11, 12] + in[13] * md_matrix[11, 13] + in[14] * md_matrix[11, 14] +
              in[15] * md_matrix[11, 15];

    out[12] = in[0] * md_matrix[12, 0] + in[1] * md_matrix[12, 1] + in[2] * md_matrix[12, 2] +
              in[3] * md_matrix[12, 3] + in[4] * md_matrix[12, 4] + in[5] * md_matrix[12, 5] +
              in[6] * md_matrix[12, 6] + in[7] * md_matrix[12, 7] + in[8] * md_matrix[12, 8] +
              in[9] * md_matrix[12, 9] + in[10] * md_matrix[12, 10] + in[11] * md_matrix[12, 11] +
              in[12] * md_matrix[12, 12] + in[13] * md_matrix[12, 13] + in[14] * md_matrix[12, 14] +
              in[15] * md_matrix[12, 15];

    out[13] = in[0] * md_matrix[13, 0] + in[1] * md_matrix[13, 1] + in[2] * md_matrix[13, 2] +
              in[3] * md_matrix[13, 3] + in[4] * md_matrix[13, 4] + in[5] * md_matrix[13, 5] +
              in[6] * md_matrix[13, 6] + in[7] * md_matrix[13, 7] + in[8] * md_matrix[13, 8] +
              in[9] * md_matrix[13, 9] + in[10] * md_matrix[13, 10] + in[11] * md_matrix[13, 11] +
              in[12] * md_matrix[13, 12] + in[13] * md_matrix[13, 13] + in[14] * md_matrix[13, 14] +
              in[15] * md_matrix[13, 15];

    out[14] = in[0] * md_matrix[14, 0] + in[1] * md_matrix[14, 1] + in[2] * md_matrix[14, 2] +
              in[3] * md_matrix[14, 3] + in[4] * md_matrix[14, 4] + in[5] * md_matrix[14, 5] +
              in[6] * md_matrix[14, 6] + in[7] * md_matrix[14, 7] + in[8] * md_matrix[14, 8] +
              in[9] * md_matrix[14, 9] + in[10] * md_matrix[14, 10] + in[11] * md_matrix[14, 11] +
              in[12] * md_matrix[14, 12] + in[13] * md_matrix[14, 13] + in[14] * md_matrix[14, 14] +
              in[15] * md_matrix[14, 15];

    out[15] = in[0] * md_matrix[15, 0] + in[1] * md_matrix[15, 1] + in[2] * md_matrix[15, 2] +
              in[3] * md_matrix[15, 3] + in[4] * md_matrix[15, 4] + in[5] * md_matrix[15, 5] +
              in[6] * md_matrix[15, 6] + in[7] * md_matrix[15, 7] + in[8] * md_matrix[15, 8] +
              in[9] * md_matrix[15, 9] + in[10] * md_matrix[15, 10] + in[11] * md_matrix[15, 11] +
              in[12] * md_matrix[15, 12] + in[13] * md_matrix[15, 13] + in[14] * md_matrix[15, 14] +
              in[15] * md_matrix[15, 15];
}

void MatrixMultiply(std::span<const float> in, std::span<float> out, std::span<const float> matrix, size_t N)
{
    assert(in.size() % N == 0 && out.size() == in.size());

    auto md_matrix = std::mdspan(matrix.data(), N, N);

    const size_t kRowCount = in.size() / N;

    for (size_t k = 0; k < kRowCount; ++k)
    {
        const size_t offset = k * N;
        for (size_t i = 0; i < N; ++i)
        {
            out[i + offset] = 0.0f;
            for (size_t j = 0; j < N; ++j)
            {
                out[i + offset] += in[j + offset] * md_matrix[i, j];
            }
        }
    }
}

} // namespace fdn