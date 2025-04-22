// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sfFDN
{
/** @defgroup DelayUtils Delay Utilities
 * @brief A collection of functions to generate delay lengths for feedback delay networks.
 * @{
 */

/**
 * @brief Generates a list of delay lengths based on the specified parameters.
 * @param delay_count the number of delay lengths to generate
 * @param min_delay the minimum delay length
 * @param max_delay the maximum delay length
 * @param type the type of delay length distribution to use
 * @param seed the random seed to use for generating delays
 * @return a vector containing the generated delay lengths
 *
 *  @note While this method uses `min_delay` and `max_delay`, it does not guarantee that the delays will be within
    this range when using DelayLengthType::SteamAudio.
 */
std::vector<float> GetDelayLengths(uint32_t delay_count, float min_delay, float max_delay, DelayLengthType type,
                                   uint32_t seed = 0);

/**
 * @brief Generates a list of delay lengths logarithmically spaced around a mean value.
 *
 * @param delay_count the number of delay lengths to generate
 * @param mean_delay_ms the mean delay length in milliseconds
 * @param sigma the standard deviation for the distribution
 * @param sample_rate the sample rate in Hz
 * @return std::vector<uint32_t> a vector containing the generated delay lengths
 */
std::vector<float> GetDelayLengthsFromMean(uint32_t delay_count, float mean_delay_ms, float sigma,
                                           uint32_t sample_rate);

/** @} */
} // namespace sfFDN