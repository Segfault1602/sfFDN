#include "rng.h"

#include "pch.h"

namespace sfFDN
{
uint32_t rng()
{
    static uint32_t y = 2463534242;
    y ^= (y << 13);
    y ^= (y >> 17);
    return (y ^= (y << 5));
}
} // namespace sfFDN