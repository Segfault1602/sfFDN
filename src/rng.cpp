#include "rng.h"

#include <cstdint>

namespace sfFDN
{
RNG::RNG(uint32_t seed)
    : state(seed)
{
}

} // namespace sfFDN