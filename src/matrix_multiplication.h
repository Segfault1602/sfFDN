#pragma once

#include <cassert>
#include <mdspan>
#include <span>

namespace fdn
{

void HadamardMultiply(const std::span<const float> input, std::span<float> output);
void WalshHadamardTransform(std::span<float> inout);

void MatrixMultiply_4(const std::span<const float, 4> input, std::span<float, 4> output,
                      const std::span<const float, 4 * 4> matrix);
void MatrixMultiply_8(const std::span<const float, 8> input, std::span<float, 8> output,
                      const std::span<const float, 8 * 8> matrix);
void MatrixMultiply_16(const std::span<const float, 16> input, std::span<float, 16> output,
                       const std::span<const float, 16 * 16> matrix);

void MatrixMultiply(std::span<const float> in, std::span<float> out, std::span<const float> matrix, size_t N);

} // namespace fdn