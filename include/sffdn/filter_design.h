// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/filter.h"
#include "sffdn/filterbank.h"
#include "sffdn/types.h"

#include <array>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace sfFDN
{

/** @defgroup FilterDesign Filter Design
 * @brief A collection of functions to design filters for feedback delay networks.
 * @{
 */

/**
 * @brief Get the coefficients of a one-pole absorption filter
 * @param config Structure containing the filter design parameters
 * @return A pair of floats where the first element is the b coefficient and the second element is the a coefficient of
 * the one-pole filter.
 * @note Based on Jot, J. M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators (pp.
 * 1-12). Presented at the Proc. Audio Eng. Soc. Conv., Paris, France.
 */
std::pair<float, float> DesignTwoBandAbsorption(const TwoBandFilterOptions& options);

/**
 * @brief Design a three-band absorption filter consisting of a low-shelf, high-shelf and a gain factor to match the
 * desired T60 at DC, mid and Nyquist frequencies.
 * @param params Structure containing the filter design parameters
 * @param delay Delay in samples for the delay line preceding the filter
 * @return std::array<FilterCoefficients, 2> Coefficients of the designed EQ filter.
 */
std::array<FilterCoefficients, 2> DesignThreeBandAbsorption(const ThreeBandFilterOptions& options);

/**
 * @brief Design an attenuation filter according to the method described in [1]
 * @param t60s Reverberation time in seconds for each band
 * @param delay Delay in samples for the delay line preceding the filter
 * @param sr Sample rate in Hz
 * @param shelf_cutoff Cutoff frequency for the low shelf filter in Hz used as the pre-filter
 * @return Coefficients of the designed EQ filter where the first 6 floats are the coefficients (b0, b1, b2, a0, a1,
 * a2) of the first filter, and the next 6 floats are the coefficients of the second filter, and so on.
 * @note [1] V. Välimäki, K. Prawda, and S. J. Schlecht, "Two-Stage Attenuation Filter for Artificial Reverberation,"
 * IEEE Signal Processing Letters, vol. 31, pp. 391–395, 2024, doi: 10.1109/LSP.2024.3352510.
 * @note Original MATLAB implementation: https://github.com/KPrawda/Two_stage_filter/blob/main/twoFilters.m
 */
std::array<FilterCoefficients, 11> DesignTenBandAbsorption(const TenBandFilterOptions& options);

/**
 * @brief Design an octave EQ filter consisting of a low shelf, high shelf and 8 band-pass peaking filters
 * @param mag Magnitude response in dB for each octave band
 * @param freqs Center frequencies of the octave bands in Hz
 * @param sr Sample rate in Hz
 * @return Coefficients of the designed EQ filter where the first 6 floats are the coefficients (b0, b1, b2, a0, a1,
 * a2) of the first filter, and the next 6 floats are the coefficients of the second filter, and so on.
 * @note The implementation is based on the method described in [1] and uses the RBJ cookbook formulas for the
 * low-shelf and high-shelf filters.
 * @note [1] V. Valimaki and J. Liski, "Accurate Cascade Graphic Equalizer," IEEE Signal Process. Lett., vol. 24, no.
 * 2, pp. 176–180, Feb. 2017, doi: 10.1109/LSP.2016.2645280.
 * @note Original MATLAB implementation: https://github.com/KPrawda/Two_stage_filter/blob/main/aceq.m
 */
std::array<FilterCoefficients, 11> DesignGraphicEQ(const GraphicEQOptions& options);

/** @brief Create an attenuation filter processor based on the provided configuration variant.
 *
 * @param config A variant containing the configuration for the attenuation filter design. The specific type of
 * filter will be determined by the type of the variant.
 * @return A unique pointer to the created AttenuationFilter processor.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilter(const attenuation_filter_variant_t& options);

/**
 * @brief Create a Attenuation Filter Bank object
 *
 * @param configs The configuration for each filter in the bank. Each filter can be of a different type, as specified by
 * the attenuation_filter_variant_t variant.
 * @return std::unique_ptr<AudioProcessor>
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const AttenuationFilterBankOptions& options);

/** @brief Creates an attenuation filter bank processor based on the provided configuration variant and delay values.
 *
 * @param variant_config A variant containing the configuration for the attenuation filter design. The specific type
 of
 * filter will be determined by the type of the variant.
 * @param delays Delay in samples for each delay line preceding the filters
 * @return A unique pointer to the created FilterBank processor containing the attenuation filters.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const attenuation_filter_variant_t& options,
                                                            std::span<const float> delays);

/** @} */
} // namespace sfFDN