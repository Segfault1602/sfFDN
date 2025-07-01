#pragma once

#include <cstddef>
#include <vector>

namespace sfFDN
{
enum class DelayLengthType
{
    Random,
    Gaussian,
    Primes,
    Uniform,
    PrimePower,
    SteamAudio,
};

std::vector<uint32_t> GetDelayLengths(uint32_t N, uint32_t min_delay, uint32_t max_delay, DelayLengthType type,
                                      uint32_t seed = 0);
} // namespace sfFDN