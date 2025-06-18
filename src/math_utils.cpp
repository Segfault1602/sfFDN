#include "math_utils.h"

namespace sfFDN
{

bool Math::IsPowerOfTwo(size_t n)
{
    return n && !(n & (n - 1));
}

size_t Math::NextPowerOfTwo(size_t n)
{
    if (n == 0)
    {
        return 1;
    }
    --n;
    for (size_t i = 1; i < sizeof(size_t) * 8; i <<= 1)
    {
        n |= n >> i;
    }
    return n + 1;
}
} // namespace sfFDN