#include "array_math.h"

#include "pch.h"

namespace sfFDN
{

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

void ArrayMath::ScaleAdd(std::span<const float> a, const float b, std::span<const float> c, std::span<float> out)
{
    assert(a.size() == out.size());
    assert(c.size() == out.size());

    for (auto [x, y, z] : std::views::zip(a, c, out))
    {
        z = x * b + y;
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
} // namespace sfFDN