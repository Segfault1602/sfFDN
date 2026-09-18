// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/filter.h"
#include "sffdn/filterbank.h"
#include "sffdn/types.h"

#include <array>
#include <memory>
#include <numbers>
#include <span>
#include <utility>

namespace sfFDN
{

/** @defgroup FilterDesign Filter Design
 * @{
 */

/** @brief RBJ low-shelf design parameters. */
struct LowShelfOptions
{
    float frequency{1000.f};                     //!< Shelf midpoint frequency in Hz, strictly between zero and Nyquist.
    float gain_db{0.f};                          //!< Low-frequency gain in dB.
    float q{1.f / std::numbers::sqrt2_v<float>}; //!< Positive quality factor.

    bool operator==(const LowShelfOptions&) const = default;
};

/** @brief RBJ high-shelf design parameters.*/
struct HighShelfOptions
{
    float frequency{1000.f};                     //!< Shelf midpoint frequency in Hz, strictly between zero and Nyquist.
    float gain_db{0.f};                          //!< High-frequency gain in dB.
    float q{1.f / std::numbers::sqrt2_v<float>}; //!< Positive quality factor.

    bool operator==(const HighShelfOptions&) const = default;
};

/** @brief RBJ cookbook peaking EQ parameters */
struct PeakingOptions
{
    float frequency{1000.f};                     //!< Center frequency in Hz, strictly between zero and Nyquist.
    float gain_db{0.f};                          //!< Gain at the center frequency in dB.
    float q{1.f / std::numbers::sqrt2_v<float>}; //!< Positive cookbook quality factor.

    bool operator==(const PeakingOptions&) const = default;
};

/** @brief Designs filter coefficients.
 *
 * Design is a setup-time operation and may allocate or throw. The rate has no setter; create another designer for
 * another processing rate. Returned coefficients and processors created with this designer do not depend on its
 * lifetime. Option delays remain in samples, T60s in seconds, and frequencies in Hz.
 */
class FilterDesigner
{
  public:
    /** @brief Sets the design sample rate in Hz.
     * @throws std::invalid_argument if the rate is not positive.
     * @pre Numeric inputs to the public C++ API are finite.
     */
    explicit FilterDesigner(float sample_rate);

    /** @brief Returns the design sample rate in Hz. */
    [[nodiscard]] float GetSampleRate() const noexcept;

    /** @brief Returns the homogeneous attenuation's linear gain. */
    [[nodiscard]] float DesignFilter(const HomogenousFilterOptions& options) const;

    /** @brief Returns one-pole coefficients (b0, a1), with y[n] = b0*x[n] - a1*y[n-1].
     * @note Based on Jot and Chaigne (1991), Digital delay networks for designing artificial reverberators.
     */
    [[nodiscard]] std::pair<float, float> DesignFilter(const TwoBandFilterOptions& options) const;

    /** @brief Returns two shelf sections matching the desired T60 at DC, mid and Nyquist frequencies. */
    [[nodiscard]] std::array<FilterCoefficients, 2> DesignFilter(const ThreeBandFilterOptions& options) const;

    /** @brief Returns eleven sections for two-stage ten-band attenuation.
     * @note Valimaki, Prawda and Schlecht (2024), doi: 10.1109/LSP.2024.3352510.
     */
    [[nodiscard]] std::array<FilterCoefficients, 11> DesignFilter(const TenBandFilterOptions& options) const;

    /** @brief Returns eleven sections for the cascade Graphic EQ.
     * @note Valimaki and Liski (2017), doi: 10.1109/LSP.2016.2645280.
     */
    [[nodiscard]] std::array<FilterCoefficients, 11> DesignFilter(const GraphicEQOptions& options) const;

    /** @brief Returns one normalized RBJ low-shelf section, with a0 equal to one. */
    [[nodiscard]] FilterCoefficients DesignFilter(const LowShelfOptions& options) const;

    /** @brief Returns one normalized RBJ high-shelf section, with a0 equal to one. */
    [[nodiscard]] FilterCoefficients DesignFilter(const HighShelfOptions& options) const;

    /** @brief Returns one normalized RBJ cookbook peaking section, with a0 equal to one. */
    [[nodiscard]] FilterCoefficients DesignFilter(const PeakingOptions& options) const;

    /** @brief Converts a positive T60 in seconds and a nonnegative delay in samples to linear amplitude gain.
     *
     * The gain is 10^(-3 * delay_samples / (t60_seconds * sample_rate)). Zero delay returns unity.
     * @throws std::invalid_argument if T60 is not positive or delay is negative.
     */
    [[nodiscard]] float T60ToGain(float t60_seconds, float delay_samples) const;

  private:
    float sample_rate_;
};

/** @brief Creates an attenuation processor from a concrete or variant filter configuration.
 * @param options Filter parameters, including the actual delay in samples.
 * @param designer The construction-time designer; the processor does not retain a reference to it.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilter(const attenuation_filter_variant_t& options,
                                                        const FilterDesigner& designer);

/** @brief Creates an attenuation bank whose channels may have different filter types.
 * @param options Per-channel filter parameters, including actual delays in samples.
 * @param designer Supplies the sample rate for every filter in the bank.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const AttenuationFilterBankOptions& options,
                                                            const FilterDesigner& designer);

/** @brief Creates an attenuation bank using one filter recipe and a delay per channel.
 * @param options The filter recipe; its delay is replaced by each channel's delay.
 * @param delays Actual delay lengths in samples.
 * @param designer Supplies the sample rate for every filter in the bank.
 */
std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const attenuation_filter_variant_t& options,
                                                            std::span<const float> delays,
                                                            const FilterDesigner& designer);

/** @} */
} // namespace sfFDN
