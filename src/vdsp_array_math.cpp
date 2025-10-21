#include "array_math.h"

#include <Accelerate/Accelerate.h>

namespace sfFDN
{
void ArrayMath::Accumulate(std::span<float> a, std::span<const float> b)
{
    vDSP_vadd(a.data(), 1, b.data(), 1, a.data(), 1, a.size());
}

void ArrayMath::Add(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    vDSP_vadd(a.data(), 1, b.data(), 1, out.data(), 1, a.size());
}

void ArrayMath::Scale(std::span<const float> a, const float b, std::span<float> out)
{
    vDSP_vsmul(a.data(), 1, &b, out.data(), 1, a.size());
}

void ArrayMath::ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out)
{
    vDSP_vsma(a.data(), 1, &b, out.data(), 1, out.data(), 1, a.size());
}

} // namespace sfFDN