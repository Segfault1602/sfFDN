#pragma once

#include <cstddef>
#include <vector>

namespace sfFDN
{
enum class DelayLengthType
{
    Random = 0,
    Gaussian = 1,
    Primes = 2,
    Uniform = 3,
    PrimePower = 4,
    SteamAudio = 5,
    Count,
};

std::vector<uint32_t> GetDelayLengths(uint32_t N, uint32_t min_delay, uint32_t max_delay, DelayLengthType type,
                                      uint32_t seed = 0);
} // namespace sfFDN