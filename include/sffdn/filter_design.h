// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "filterbank.h"

namespace sfFDN
{
/** @defgroup FilterDesign Filter Design
 * @brief A collection of functions to design filters for feedback delay networks.
 * @{
 */

/**
 * @brief Get the coefficients of a one-pole absorption filter
 * @param t60_dc Reverberation time in seconds at DC (0 Hz)
 * @param t60_ny Reverberation time in seconds at Nyquist frequency
 * @param sr Sample rate in Hz
 * @param delay Delay in samples for the delay line preceding the filter
 * @param b Output parameter for the b0 coefficient of the filter
 * @param a Output parameter for the a1 coefficient of the filter
 * @note Based on Jot, J. M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators (pp.
 * 1-12). Presented at the Proc. Audio Eng. Soc. Conv., Paris, France.
 */
void GetOnePoleAbsorption(float t60_dc, float t60_ny, float sr, float delay, float& b, float& a);

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
std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff = 8000.0f);

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
std::vector<float> DesignGraphicEQ(std::span<const float> mag, std::span<const float> freqs, float sr);

/** @brief Creates an attenuation filter bank using the two-stage attenuation filter design.
 *
 * @param t60s Reverberation time in seconds for each band
 * @param delays Delay in samples for each delay line preceding the filter
 * @param sample_rate Sample rate in Hz
 * @return A unique pointer to the created FilterBank processor containing the attenuation filters.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(std::span<const float> t60s,
                                                            std::span<const uint32_t> delays, float sample_rate);

/** @} */
} // namespace sfFDN