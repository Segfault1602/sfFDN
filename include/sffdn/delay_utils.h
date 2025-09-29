// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sfFDN
{

/// @brief Types of delay length distributions.
enum class DelayLengthType : uint8_t
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
    Count = 6,
};

/// @brief Generates a list of delay lengths based on the specified parameters.
/// @param channel_count the number of delay lengths to generate
/// @param min_delay the minimum delay length
/// @param max_delay the maximum delay length
/// @param type the type of delay length distribution to use
/// @param seed the random seed to use for generating delays
/// @return a vector containing the generated delay lengths
std::vector<uint32_t> GetDelayLengths(uint32_t channel_count, uint32_t min_delay, uint32_t max_delay,
                                      DelayLengthType type, uint32_t seed = 0);

std::vector<uint32_t> GetDelayLengthsFromMean(uint32_t channel_count, float mean_delay_ms, float sigma,
                                              uint32_t sample_rate);
} // namespace sfFDN

/**
 * @brief Utility functions for delay calculations.
 */
namespace DelayUtils
{
/**
 * @brief Convert delay time from milliseconds to samples.
 * @param delay_ms Delay time in milliseconds.
 * @param sample_rate Sample rate in Hz.
 * @return Delay time in samples.
 */
uint32_t MsToSamples(float delay_ms, float sample_rate);

/**
 * @brief Convert delay time from samples to milliseconds.
 * @param delay_samples Delay time in samples.
 * @param sample_rate Sample rate in Hz.
 * @return Delay time in milliseconds.
 */
float SamplesToMs(uint32_t delay_samples, float sample_rate);

/**
 * @brief Generate optimal delay times for FDN.
 * @param num_delays Number of delay lines.
 * @param min_delay_ms Minimum delay time in milliseconds.
 * @param max_delay_ms Maximum delay time in milliseconds.
 * @param sample_rate Sample rate in Hz.
 * @return Vector of delay times in samples.
 */
std::vector<uint32_t> GenerateOptimalDelays(uint32_t num_delays, float min_delay_ms, float max_delay_ms,
                                            float sample_rate);
} // namespace DelayUtils