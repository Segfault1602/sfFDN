#pragma once

#include <span>

namespace fdn
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

// Scales an array by a constant and adds another array
// out[i] = a[i] * b + c[i]
void ScaleAdd(std::span<const float> a, const float b, std::span<const float> c, std::span<float> out);

void ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out);

} // namespace ArrayMath

} // namespace fdn
