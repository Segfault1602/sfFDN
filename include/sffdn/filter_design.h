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
std::pair<float, float> GetOnePoleAbsorption(const TwoBandFilterConfig& config);

struct ThreeBandAbsorptionParams
{
    float t60_dc;
    float t60_mid;
    float t60_ny;
    float low_shelf_cutoff = 300.f;
    float high_shelf_cutoff = 8000.f;
    float q = 1.f / std::numbers::sqrt2_v<float>;
    float sample_rate;
};

/**
 * @brief Design a three-band absorption filter consisting of a low-shelf, high-shelf and a gain factor to match the
 * desired T60 at DC, mid and Nyquist frequencies.
 * @param params Structure containing the filter design parameters
 * @param delay Delay in samples for the delay line preceding the filter
 * @return std::array<FilterCoefficients, 2> Coefficients of the designed EQ filter.
 */
std::array<FilterCoefficients, 2> DesignThreeBandAbsorption(const ThreeBandFilterConfig& params);

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
std::array<FilterCoefficients, 11> GetTwoFilter(const TenBandFilterConfig& config);

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
std::array<FilterCoefficients, 11> DesignGraphicEQ(const GraphicEQConfig& config);

/** @brief Creates an attenuation filter bank using the two-stage attenuation filter design.
 *
 * @param t60s Reverberation time in seconds for each band
 * @param delays Delay in samples for each delay line preceding the filter
 * @param sample_rate Sample rate in Hz
 * @return A unique pointer to the created FilterBank processor containing the attenuation filters.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(attenuation_filter_variant_t variant_config,
                                                            std::span<const float> delays, float sample_rate);

std::unique_ptr<AudioProcessor> CreateAttenuationFilter(const attenuation_filter_variant_t& config);
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const AttenuationFilterBankConfig& configs);
/** @} */
} // namespace sfFDN