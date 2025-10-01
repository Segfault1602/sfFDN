// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delay.h"

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

    SchroederAllpass(SchroederAllpass&&) = default;
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
    float Tick(float input);

    /** @brief Processes a block of samples through the filter.
     * @param in The input samples.
     * @param out The output samples.
     * The input and output spans must have the same size.
     */
    void ProcessBlock(std::span<const float> in, std::span<float> out);

    /** @brief Clears the filter state.
     * This sets the internal delay buffer to zero.
     */
    void Clear();

  private:
    Delay delay_;
    float g_{};
};

/** @brief A section of Schroeder allpass filters in series */
class SchroederAllpassSection : public AudioProcessor
{
  public:
    /** @brief Constructs an empty SchroederAllpassSection. */
    SchroederAllpassSection() = default;

    /** @brief Constructs a SchroederAllpassSection with a given number of filters.
     * @param filter_count The number of allpass filters in the section.
     */
    SchroederAllpassSection(uint32_t filter_count);

    SchroederAllpassSection(const SchroederAllpassSection&) = delete;
    SchroederAllpassSection& operator=(const SchroederAllpassSection&) = delete;

    SchroederAllpassSection(SchroederAllpassSection&&) noexcept;
    SchroederAllpassSection& operator=(SchroederAllpassSection&&) noexcept;

    ~SchroederAllpassSection() = default;

    /** @brief Sets the number of allpass filters in the section.
     * @param filter_count The number of allpass filters.
     */
    void SetFilterCount(uint32_t filter_count);

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
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Gets the number of input channels supported.
     * This is always 1, as SchroederAllpassSection processes one channel at a time.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Gets the number of output channels supported.
     * This is always 1, as SchroederAllpassSection processes one channel at a time.
     */
    uint32_t OutputChannelCount() const override;

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
};

/** \brief Implements parallel Schroeder allpass sections. */
class ParallelSchroederAllpassSection : public AudioProcessor
{
  public:
    /** \brief Constructs a ParallelSchroederAllpassSection with a given number of channels and stages.
     * @param channel_count The number of parallel channels.
     * @param stage_count The number of allpass filters in each channel.
     */
    ParallelSchroederAllpassSection(uint32_t channel_count, uint32_t stage_count);

    /** @brief Sets the delays for each allpass filter in the section.
     * @param delays A span of delay values in samples.
     * The size of the span must be equal to (channel_count * stage_count).
     */
    void SetDelays(std::span<const uint32_t> delays);

    /** @brief Sets the feedback gains for each allpass filter in the section.
     * @param gains A span of feedback gain values.
     * The size of the span must be equal to (channel_count * stage_count).
     */
    void SetGains(std::span<const float> gains);

    /** @brief Processes a block of audio through the section.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples and channels equal to InputChannelCount().
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Gets the number of input channels supported.*/
    uint32_t InputChannelCount() const override;

    /** @brief Gets the number of output channels supported.*/
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of all allpass filters in the section.
     */
    void Clear() override;

    /** @brief Creates a copy of the ParallelSchroederAllpassSection.
     * @return A unique pointer to the cloned ParallelSchroederAllpassSection.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<SchroederAllpassSection> allpasses_;
    uint32_t stage_count_;
};
} // namespace sfFDN