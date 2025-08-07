#include "array_math.h"

#include <cassert>
#include <cstdint>

namespace sfFDN
{

void ArrayMath::Accumulate(std::span<float> a, std::span<const float> b)
{
    assert(a.size() == b.size());

    const uint32_t unroll_size = a.size() & ~3;
    int idx = 0;
    while (idx < unroll_size)
    {
        a[idx + 0] += b[idx + 0];
        a[idx + 1] += b[idx + 1];
        a[idx + 2] += b[idx + 2];
        a[idx + 3] += b[idx + 3];
        idx += 4;
    }

    while (idx < a.size())
    {
        a[idx] += b[idx];
        ++idx;
    }
}

void ArrayMath::Add(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    const uint32_t unroll_size = a.size() & ~3;
    int idx = 0;
    while (idx < unroll_size)
    {
        out[idx + 0] = a[idx + 0] + b[idx + 0];
        out[idx + 1] = a[idx + 1] + b[idx + 1];
        out[idx + 2] = a[idx + 2] + b[idx + 2];
        out[idx + 3] = a[idx + 3] + b[idx + 3];
        idx += 4;
    }
    while (idx < a.size())
    {
        out[idx] = a[idx] + b[idx];
        ++idx;
    }
}

void ArrayMath::Scale(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    const uint32_t unroll_size = a.size() & ~3;
    int idx = 0;
    while (idx < unroll_size)
    {
        out[idx + 0] = a[idx + 0] * b;
        out[idx + 1] = a[idx + 1] * b;
        out[idx + 2] = a[idx + 2] * b;
        out[idx + 3] = a[idx + 3] * b;
        idx += 4;
    }

    while (idx < a.size())
    {
        out[idx] = a[idx] * b;
        ++idx;
    }
}

void ArrayMath::ScaleAdd(std::span<const float> a, const float b, std::span<const float> c, std::span<float> out)
{
    assert(a.size() == c.size());
    assert(a.size() == out.size());

    const uint32_t unroll_size = a.size() & ~3;
    int idx = 0;
    while (idx < unroll_size)
    {
        out[idx + 0] = a[idx + 0] * b + c[idx + 0];
        out[idx + 1] = a[idx + 1] * b + c[idx + 1];
        out[idx + 2] = a[idx + 2] * b + c[idx + 2];
        out[idx + 3] = a[idx + 3] * b + c[idx + 3];
        idx += 4;
    }

    while (idx < a.size())
    {
        out[idx] = a[idx] * b + c[idx];
        ++idx;
    }
}

void ArrayMath::ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    const uint32_t unroll_size = a.size() & ~3;
    int idx = 0;
    while (idx < unroll_size)
    {
        out[idx + 0] += a[idx + 0] * b;
        out[idx + 1] += a[idx + 1] * b;
        out[idx + 2] += a[idx + 2] * b;
        out[idx + 3] += a[idx + 3] * b;
        idx += 4;
    }

    while (idx < a.size())
    {
        out[idx] += a[idx] * b;
        ++idx;
    }
}
} // namespace sfFDN