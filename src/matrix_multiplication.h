#pragma once

#include "sffdn/attributes.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>

namespace sfFDN
{

class AudioBuffer;

void HadamardMultiply(std::span<const float> input, std::span<float> output);
void WalshHadamardTransform(std::span<float> inout);

void HadamardMultiplyBlock(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;
void HouseholderMultiplyBlock(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;

void MatrixMultiply_4(std::span<const float, 4> input, std::span<float, 4> output,
                      std::span<const float, 4 * 4> matrix);
void MatrixMultiply_8(std::span<const float, 8> input, std::span<float, 8> output,
                      std::span<const float, 8 * 8> matrix);
void MatrixMultiply_16(std::span<const float, 16> input, std::span<float, 16> output,
                       std::span<const float, 16 * 16> matrix);

/// @brief Matrix multiplication
/// @param in k x mat_size input matrix in column-major order
/// @param out k x mat_size output matrix in column-major order
/// @param matrix mat_size x mat_size transformation matrix in column-major order
/// @param mat_size size of the transformation matrix (mat_size x mat_size)
void MatrixMultiply_C(std::span<const float> in, std::span<float> out, std::span<const float> matrix,
                      uint32_t mat_size);

template <uint32_t N>
void FWHT(std::span<float, N> data)
{
    for (auto h = 1u; h < N; h *= 2)
    {
        for (auto i = 0u; i < N; i += 2 * h)
        {
            for (auto j = 0u; j < h; ++j)
            {
                const float a = data[i + j];
                const float b = data[i + j + h];
                data[i + j] = a + b;
                data[i + j + h] = a - b;
            }
        }
    }

    const float normalization_factor = 1.f / std::sqrt(static_cast<float>(N));
    for (float& i : data)
    {
        i *= normalization_factor;
    }
}

} // namespace sfFDN