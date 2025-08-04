// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <vector>

namespace sfFDN
{

/// @brief Types of delay length distributions.
enum class DelayLengthType
{
    /// @brief Delay lengths are generated randomly within the specified range based on a uniform distribution.
    Random = 0,
    /// @brief Delay lengths are generated based on a Gaussian distribution within the specified range.
    Gaussian = 1,
    /// @brief Delay lengths are selected randomly from a list of prime numbers.
    Primes = 2,
    /// @brief Delay lengths are uniformly distributed within the specified range.
    Uniform = 3,
    /// @brief Delay lengths are generated as powers of prime numbers within the specified range.
    /// Based on https://ccrma.stanford.edu/~jos/pasp/Prime_Power_Delay_Line_Lengths.html
    PrimePower = 4,
    /// @brief Delay lengths are generated using the algorithm from the SteamAudio library.
    /// @note While this method uses `min_delay` and `max_delay`, it does not guarantee that the delays will be within
    /// this range.
    SteamAudio = 5,

    /// @brief The total number of delay length types.
    Count,
};

/// @brief Generates a list of delay lengths based on the specified parameters.
/// @param N the number of delay lengths to generate
/// @param min_delay the minimum delay length
/// @param max_delay the maximum delay length
/// @param type the type of delay length distribution to use
/// @param seed the random seed to use for generating delays
/// @return a vector containing the generated delay lengths
std::vector<uint32_t> GetDelayLengths(uint32_t N, uint32_t min_delay, uint32_t max_delay, DelayLengthType type,
                                      uint32_t seed = 0);
} // namespace sfFDN