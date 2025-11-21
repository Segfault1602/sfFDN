#include "array_math.h"

#include <cassert>
#include <cmath>
#include <ranges>
#include <span>

#ifdef SFFDN_USE_IPP
#include <ipp.h>
#endif

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

namespace sfFDN
{
#ifdef SFFDN_USE_IPP
void ArrayMath::Accumulate(std::span<float> a, std::span<const float> b)
{
    assert(a.size() == b.size());

    ippsAdd_32f_I(b.data(), a.data(), static_cast<int>(a.size()));
}

void ArrayMath::Add(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    ippsAdd_32f(a.data(), b.data(), out.data(), static_cast<int>(a.size()));
}

void ArrayMath::Scale(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    ippsMulC_32f(a.data(), b, out.data(), static_cast<int>(a.size()));
}

void ArrayMath::ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    ippsAddProductC_32f(a.data(), b, out.data(), static_cast<int>(a.size()));
}

void ArrayMath::Multiply(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    ippsMul_32f(a.data(), b.data(), out.data(), static_cast<int>(a.size()));
}

void ArrayMath::MultiplyAdd(std::span<const float> a, float b, std::span<const float> c, std::span<float> out)
{
    assert(a.size() == c.size());
    assert(a.size() == out.size());

    ippsMulC_32f(a.data(), b, out.data(), static_cast<int>(a.size()));
    ippsAdd_32f_I(c.data(), out.data(), static_cast<int>(a.size()));
}

#elifdef SFFDN_USE_VDSP
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

void ArrayMath::MultiplyAdd(std::span<const float> a, float b, std::span<const float> c, std::span<float> out)
{
    vDSP_vsma(a.data(), 1, &b, c.data(), 1, out.data(), 1, a.size());
}
#else
void ArrayMath::Accumulate(std::span<float> a, std::span<const float> b)
{
    assert(a.size() == b.size());

    for (auto [x, y] : std::views::zip(a, b))
    {
        x += y;
    }
}

void ArrayMath::Add(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    for (auto [x, y, z] : std::views::zip(a, b, out))
    {
        z = x + y;
    }
}

void ArrayMath::Scale(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    for (auto [x, y] : std::views::zip(a, out))
    {
        y = x * b;
    }
}

void ArrayMath::ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    for (auto [x, y] : std::views::zip(a, out))
    {
        y += x * b;
    }
}

void ArrayMath::Multiply(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    for (auto [a_val, b_val, out_val] : std::views::zip(a, b, out))
    {
        out_val = a_val * b_val;
    }
}

void ArrayMath::MultiplyAdd(std::span<const float> a, float b, std::span<const float> c, std::span<float> out)
{
    assert(a.size() == c.size());
    assert(a.size() == out.size());

    for (auto [a_val, c_val, out_val] : std::views::zip(a, c, out))
    {
        out_val = a_val * b + c_val;
    }
}
#endif // SFFDN_USE_IPP

} // namespace sfFDN