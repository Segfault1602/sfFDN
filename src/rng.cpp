#include "rng.h"

#include "pch.h"
#include <sys/types.h>

namespace sfFDN
{
RNG::RNG(uint32_t seed)
    : state(seed)
{
}

} // namespace sfFDN