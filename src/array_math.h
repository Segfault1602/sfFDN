#pragma once

#include <span>

namespace sfFDN
{
namespace ArrayMath
{

// Add two arrays element-wise, store the result in the first array
// a[i] += b[i]
void Accumulate(std::span<float> a, std::span<const float> b);

// Adds two arrays element-wise
// out[i] = a[i] + b[i]
void Add(std::span<const float> a, std::span<const float> b, std::span<float> out);

// Scales an array by a constant
// out[i] = a[i] * b
void Scale(std::span<const float> a, const float b, std::span<float> out);

// out[i] += a[i] * b
void ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out);

// out[i] = a[i] * b[i]
void Multiply(std::span<const float> a, std::span<const float> b, std::span<float> out);

// out[i] = a[i]*b + c[i]
void MultiplyAdd(std::span<const float> a, float b, std::span<const float> c, std::span<float> out);

} // namespace ArrayMath

} // namespace sfFDN
