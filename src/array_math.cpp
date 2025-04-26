#include "array_math.h"

#include <cassert>

namespace fdn
{

void ArrayMath::Accumulate(std::span<float> a, std::span<const float> b)
{
    assert(a.size() == b.size());

    size_t block_count = a.size() / 4;
    float* a_ptr = a.data();
    const float* b_ptr = b.data();

    while (block_count > 0)
    {
        a_ptr[0] += b_ptr[0];
        a_ptr[1] += b_ptr[1];
        a_ptr[2] += b_ptr[2];
        a_ptr[3] += b_ptr[3];
        a_ptr += 4;
        b_ptr += 4;
        --block_count;
    }
    size_t remainder = a.size() % 4;
    while (remainder > 0)
    {
        *a_ptr++ += *b_ptr++;
        --remainder;
    }
}

void ArrayMath::Add(std::span<const float> a, std::span<const float> b, std::span<float> out)
{
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    size_t block_count = a.size() / 4;
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* out_ptr = out.data();

    while (block_count > 0)
    {
        out_ptr[0] = a_ptr[0] + b_ptr[0];
        out_ptr[1] = a_ptr[1] + b_ptr[1];
        out_ptr[2] = a_ptr[2] + b_ptr[2];
        out_ptr[3] = a_ptr[3] + b_ptr[3];
        out_ptr += 4;
        a_ptr += 4;
        b_ptr += 4;
        --block_count;
    }
    size_t remainder = a.size() % 4;
    while (remainder > 0)
    {
        *out_ptr++ = *a_ptr++ + *b_ptr++;
        --remainder;
    }
}

void ArrayMath::Scale(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    size_t block_count = a.size() / 4;
    const float* a_ptr = a.data();
    float* out_ptr = out.data();

    while (block_count > 0)
    {
        out_ptr[0] = a_ptr[0] * b;
        out_ptr[1] = a_ptr[1] * b;
        out_ptr[2] = a_ptr[2] * b;
        out_ptr[3] = a_ptr[3] * b;
        out_ptr += 4;
        a_ptr += 4;
        --block_count;
    }

    size_t remainder = a.size() % 4;
    while (remainder > 0)
    {
        *out_ptr++ = *a_ptr++ * b;
        --remainder;
    }
}

void ArrayMath::ScaleAdd(std::span<const float> a, const float b, std::span<const float> c, std::span<float> out)
{
    assert(a.size() == c.size());
    assert(a.size() == out.size());

    size_t block_count = a.size() / 4;
    const float* a_ptr = a.data();
    const float* c_ptr = c.data();
    float* out_ptr = out.data();
    while (block_count > 0)
    {
        out_ptr[0] = a_ptr[0] * b + c_ptr[0];
        out_ptr[1] = a_ptr[1] * b + c_ptr[1];
        out_ptr[2] = a_ptr[2] * b + c_ptr[2];
        out_ptr[3] = a_ptr[3] * b + c_ptr[3];
        out_ptr += 4;
        a_ptr += 4;
        c_ptr += 4;
        --block_count;
    }

    size_t remainder = a.size() % 4;
    while (remainder > 0)
    {
        *out_ptr++ = *a_ptr++ * b + *c_ptr++;
        --remainder;
    }
    assert(out_ptr == out.data() + out.size());
}

void ArrayMath::ScaleAccumulate(std::span<const float> a, const float b, std::span<float> out)
{
    assert(a.size() == out.size());

    size_t block_count = a.size() / 4;
    const float* a_ptr = a.data();
    float* out_ptr = out.data();

    while (block_count > 0)
    {
        out_ptr[0] += a_ptr[0] * b;
        out_ptr[1] += a_ptr[1] * b;
        out_ptr[2] += a_ptr[2] * b;
        out_ptr[3] += a_ptr[3] * b;
        out_ptr += 4;
        a_ptr += 4;
        --block_count;
    }

    size_t remainder = a.size() % 4;
    while (remainder > 0)
    {
        *out_ptr++ += *a_ptr++ * b;
        --remainder;
    }
}
} // namespace fdn