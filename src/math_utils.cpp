#include "math_utils.h"

#include <bit>
#include <cassert>
#include <cstdint>

namespace sfFDN
{

bool Math::IsPowerOfTwo(uint32_t n)
{
    return std::has_single_bit(n);
}

uint32_t Math::NextPowerOfTwo(uint32_t n)
{
    if (n == 0)
    {
        return 1;
    }
    --n;
    for (auto i = 1; i < sizeof(uint32_t) * 8; i <<= 1)
    {
        n |= n >> i;
    }
    return n + 1;
}
} // namespace sfFDN