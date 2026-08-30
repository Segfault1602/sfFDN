// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"
#include "sffdn/filterbank.h"
#include "sffdn/oscillator.h"
#include "sffdn/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace sfFDN
{

/** @brief A single Schroeder allpass filter.
 * The allpass filter is implemented by cascading a feedback comb filter and a feedforward comb filter.
 * @ingroup AudioProcessors
 */
class SchroederAllpass
{
  public:
    /** @brief Constructs a SchroederAllpass filter. */
    SchroederAllpass() = default;

    /** @brief Constructs a SchroederAllpass filter.
     * @param delay The delay in samples.
     * @param g The feedback gain.
     */
    SchroederAllpass(uint32_t delay, float g);

    SchroederAllpass(const SchroederAllpass&) = delete;
    SchroederAllpass& operator=(const SchroederAllpass&) = delete;

    /** @brief Move constructor for the SchroederAllpass filter.
     */
    SchroederAllpass(SchroederAllpass&&) = default;

    /** @brief Move assignment operator for the SchroederAllpass filter.
     * @return A reference to the assigned SchroederAllpass filter.
     */
    SchroederAllpass& operator=(SchroederAllpass&&) = default;

    ~SchroederAllpass() = default;

    /** @brief Sets the delay in samples.
     * @param delay The delay in samples.
     */
    void SetDelay(uint32_t delay);

    /** @brief Sets the feedback gain.
     * @param g The feedback gain.
     */
    void SetG(float g);

    /** @brief Gets the current delay in samples. */
    uint32_t GetDelay() const
    {
        return delay_.GetDelay();
    }

    /** @brief Gets the filter gain. */
    float GetG() const
    {
        return g_;
    }

    /** @brief Processes a single sample through the filter.
     * @param input The input sample.
     * @return The output sample.
     */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of samples through the filter.
     * @param in The input samples.
     * @param out The output samples.
     * The input and output spans must have the same size.
     */
    void ProcessBlock(std::span<const float> in, std::span<float> out) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of samples through the filter and accumulates the output.
     * @param in The input samples.
     * @param out The output samples.
     * The input and output spans must have the same size.
     */
    void ProcessBlockAccumulate(std::span<const float> in, std::span<float> out) noexcept SFFDN_NONBLOCKING;

    /** @brief Clears the filter state.
     * This sets the internal delay buffer to zero.
     */
    void Clear();

  private:
    Delay delay_;
    float g_{};

    void Tick8(std::span<const float, 8> in, std::span<float, 8> out) noexcept SFFDN_NONBLOCKING;
};

/** @brief An energy-preserving Schroeder allpass with a fixed delay and optionally modulated gain.
 *
 * The gain-varying path uses the normalized Type V recurrence:
 * \f[
 * y = c w - g x,\quad u = c x + g w,\quad c = \sqrt{1 - g^2}.
 * \f]
 * The delay remains an integer number of samples, so every valid instantaneous gain applies an orthogonal
 * transformation to the input and delayed sample.
 *
 * @ingroup AudioProcessors
 */
class TimeVaryingSchroederAllpass
{
  public:
    /** @brief Constructs an allpass with sinusoidal gain modulation.
     *
     * @note The modulation frequency and amplitude must be non-zero, and
     * `abs(gain) + abs(modulation.amplitude)` must be strictly less than one.
     */
    TimeVaryingSchroederAllpass(uint32_t delay, float gain, const ModulationOptions& modulation);

    /** @brief Sets the fixed integer delay in samples.
     * @throws std::invalid_argument if `delay` is zero.
     */
    void SetDelay(uint32_t delay);

    /** @brief Returns the delay in samples. */
    uint32_t GetDelay() const noexcept SFFDN_NONBLOCKING
    {
        return delay_.GetDelay();
    }

    /** @brief Returns the base gain. */
    float GetG() const noexcept SFFDN_NONBLOCKING
    {
        return gain_;
    }

    /** @brief Processes one sample using the configured sinusoidally modulated gain. */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes one sample with an explicit valid gain in the open interval (-1, 1).
     *
     * This overload is useful for externally scheduled gain trajectories. It does not alter the configured
     * modulation state.
     */
    float Tick(float input, float gain) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block using the configured sinusoidally modulated gain. */
    void ProcessBlock(std::span<const float> in, std::span<float> out) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block and accumulates the result in `out`. */
    void ProcessBlockAccumulate(std::span<const float> in, std::span<float> out) noexcept SFFDN_NONBLOCKING;

    /** @brief Clears the delay state and restores the configured modulation phase. */
    void Clear();

  private:
    Delay delay_;
    SineWave lfo_;
    float gain_;
};

/** @brief A section of Schroeder allpass filters in series */
class SchroederAllpassSection : public AudioProcessor
{
  public:
    /** @brief Constructs an empty SchroederAllpassSection. */
    SchroederAllpassSection() = default;

    /** @brief Constructs a SchroederAllpassSection with a given configuration.
     * @param config The configuration for the SchroederAllpassSection.
     */
    SchroederAllpassSection(const SchroederAllpassSectionOptions& config);

    /** @brief Constructs a SchroederAllpassSection with a given number of filters.
     * @param filter_count The number of allpass filters in the section.
     */
    SchroederAllpassSection(uint32_t filter_count);

    SchroederAllpassSection(const SchroederAllpassSection&) = delete;
    SchroederAllpassSection& operator=(const SchroederAllpassSection&) = delete;

    /** @brief Move constructor for the SchroederAllpassSection.
     */
    SchroederAllpassSection(SchroederAllpassSection&& other) noexcept;

    /** @brief Move assignment operator for the SchroederAllpassSection.
     * @return A reference to the assigned SchroederAllpassSection.
     */
    SchroederAllpassSection& operator=(SchroederAllpassSection&& other) noexcept;

    ~SchroederAllpassSection() override = default;

    /** @brief Sets the number of allpass filters in the section.
     * @param filter_count The number of allpass filters.
     */
    void SetFilterCount(uint32_t filter_count);

    /** @brief Sets whether the allpass filters in the section are processed in parallel.
     * @param parallel Whether to process the filters in parallel.
     */
    void SetParallel(bool parallel);

    /** @brief Sets the delays for each allpass filter in the section.
     * @param delays A span of delay values in samples.
     * The size of the span must be equal to the number of filters in the section.
     */
    void SetDelays(std::span<const uint32_t> delays);

    /** @brief Sets the feedback gains for each allpass filter in the section.
     * @param gains A span of feedback gain values.
     * The size of the span must be equal to the number of filters in the section.
     */
    void SetGains(std::span<const float> gains);

    /** @brief Sets the feedback gain for all allpass filters in the section.
     * @param gain The feedback gain value.
     */
    void SetGain(float gain);

    /** @brief Gets the current delays for each allpass filter in the section. */
    std::vector<uint32_t> GetDelays() const;

    /** @brief Gets the current feedback gains for each allpass filter in the section. */
    std::vector<float> GetGains() const;

    /** @brief Processes a block of audio through the section.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples and channels equal to 1.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the number of input channels supported.
     * This is always 1, as SchroederAllpassSection processes one channel at a time.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the number of output channels supported.
     * This is always 1, as SchroederAllpassSection processes one channel at a time.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of all allpass filters in the section.
     */
    void Clear() override;

    /** @brief Creates a copy of the SchroederAllpassSection.
     * @return A unique pointer to the cloned SchroederAllpassSection.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<SchroederAllpass> allpasses_;
    bool parallel_ = false;
};

std::unique_ptr<FilterBank> MakeMultichannelSchroederAllpassSection(
    const MultichannelSchroederAllpassSectionOptions& options);

/** @brief A single-channel section of energy-preserving time-varying Schroeder allpasses. */
class TimeVaryingSchroederAllpassSection : public AudioProcessor
{
  public:
    /** @brief Constructs a section from its stage configurations.
     * @throws std::invalid_argument if stage counts, delays, gains, or modulation ranges are invalid.
     */
    explicit TimeVaryingSchroederAllpassSection(const TimeVaryingSchroederAllpassSectionOptions& config);

    TimeVaryingSchroederAllpassSection(const TimeVaryingSchroederAllpassSection&) = default;
    TimeVaryingSchroederAllpassSection& operator=(const TimeVaryingSchroederAllpassSection&) = default;
    TimeVaryingSchroederAllpassSection(TimeVaryingSchroederAllpassSection&&) noexcept = default;
    TimeVaryingSchroederAllpassSection& operator=(TimeVaryingSchroederAllpassSection&&) noexcept = default;
    ~TimeVaryingSchroederAllpassSection() override = default;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    void Clear() override;
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<TimeVaryingSchroederAllpass> allpasses_;
    bool parallel_{false};
};

/** @brief Builds a multichannel bank with one time-varying Schroeder allpass section per channel. */
std::unique_ptr<FilterBank> MakeMultichannelTimeVaryingSchroederAllpassSection(
    const MultichannelTimeVaryingSchroederAllpassSectionOptions& options);

} // namespace sfFDN