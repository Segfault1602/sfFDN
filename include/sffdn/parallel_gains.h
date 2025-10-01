// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "audio_processor.h"
#include "oscillator.h"

namespace sfFDN
{

/** @brief Enumeration for parallel gain processing modes. */
enum class ParallelGainsMode : uint8_t
{
    Split,   /** < Process input as a single channel and output to multiple channels */
    Merge,   /** < Process each input channel separately and output to one channel */
    Parallel /** < Process each input channel separately and output to the same number of channels */
};

/** @brief A parallel gains processor.
 * Supports three modes:
 * - Split: Single input channel, multiple output channels. Each output channel is equal to the input channel
 * scaled by a gain.
 * - Merge: Multiple input channels, single output channel. The output channel is equal to the sum of all input
 * channels scaled by their respective gains.
 * - Parallel: Multiple input channels, multiple output channels. Each output channel is equal to the corresponding
 * input channel scaled by a gain.
 * @ingroup AudioProcessors
 */
class ParallelGains : public AudioProcessor
{
  public:
    /** @brief Constructs a ParallelGains processor.
     * @param mode The processing mode to use.
     */
    ParallelGains(ParallelGainsMode mode);

    /** @brief Constructs a ParallelGains processor.
     * @param channel_count The number of channels to process.
     * @param mode The processing mode to use.
     * @param gain The gain to apply.
     */
    ParallelGains(uint32_t channel_count, ParallelGainsMode mode, float gain = 1.0f);

    /** @brief Constructs a ParallelGains processor.
     * @param mode The processing mode to use.
     * @param gains A span of gains to apply to each channel.
     */
    ParallelGains(ParallelGainsMode mode, std::span<const float> gains);

    /** @brief Sets the processing mode.
     * @param mode The processing mode to use.
     */
    void SetMode(ParallelGainsMode mode);

    /** @brief Sets the gains for each channel.
     * @param gains A span of gains to apply to each channel.
     * The size of the span must be equal to InputChannelCount() for ParallelGainsMode::Merge.
     * The size of the span must be equal to OutputChannelCount() for ParallelGainsMode::Split.
     * The size of the span must be equal to InputChannelCount() and OutputChannelCount() for
     * ParallelGainsMode::Parallel.
     */
    void SetGains(std::span<const float> gains);

    /** @brief Gets the gains for each channel.
     * @param gains A span to store the gains.
     * The size of the span must be equal to the number of gains set in SetGains().
     */
    void GetGains(std::span<float> gains) const;

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples.
     * For ParallelGainsMode::Split, the input buffer must have 1 channels and the output
     * buffer must have OutputChannelCount() channels.
     * For ParallelGainsMode::Merge, the input buffer must have InputChannelCount() channels and the output buffer must
     * have 1 channels. For ParallelGainsMode::Parallel, the input and output buffers must have the same number of
     * channels equal to InputChannelCount() and OutputChannelCount().
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Gets the number of input channels supported.
     * @note For ParallelGainsMode::Split, this is always 1.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Gets the number of output channels supported.
     * @note For ParallelGainsMode::Merge, this is always 1.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function does nothing as there is no internal state to clear.
     */
    void Clear() override;

    /** @brief Creates a copy of the ParallelGains processor.
     * @return A unique pointer to the cloned ParallelGains processor.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    void ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockParallel(const AudioBuffer& input, AudioBuffer& output);

    std::vector<float> gains_;
    ParallelGainsMode mode_;
};

/** @brief A time-varying parallel gains processor.
 * Similar to ParallelGains but with time-varying gains modulated by LFOs.
 * @ingroup AudioProcessors
 */
class TimeVaryingParallelGains : public AudioProcessor
{
  public:
    /**
     * @brief Constructs a TimeVaryingParallelGains processor.
     * @param mode The processing mode to use.
     */
    TimeVaryingParallelGains(ParallelGainsMode mode);

    /** @brief Constructs a TimeVaryingParallelGains processor.
     * @param channel_count The number of channels to process.
     * @param mode The processing mode to use.
     * @param gain The initial gain to apply to each channel.
     */
    TimeVaryingParallelGains(uint32_t channel_count, ParallelGainsMode mode, float gain = 1.0f);

    /** @brief Constructs a TimeVaryingParallelGains processor.
     * @param mode The processing mode to use.
     * @param gains A span of initial gains to apply to each channel.
     */
    TimeVaryingParallelGains(ParallelGainsMode mode, std::span<const float> gains);

    /** @brief Sets the processing mode.
     * @param mode The processing mode to use.
     */
    void SetMode(ParallelGainsMode mode);

    /** @brief Sets the center gains for each channel.
     * @param gains A span of center gains to apply to each channel.
     * The size of the span must be equal to InputChannelCount() for ParallelGainsMode::Merge.
     * The size of the span must be equal to OutputChannelCount() for ParallelGainsMode::Split.
     * The size of the span must be equal to InputChannelCount() and OutputChannelCount() for
     * ParallelGainsMode::Parallel.
     */
    void SetCenterGains(std::span<const float> gains);

    /** @brief Gets the center gains for each channel.
     * @param gains A span to store the gains.
     * The size of the span must be equal to the number of gains set in SetGains().
     */
    void GetCenterGains(std::span<float> gains) const;

    /** @brief Sets the LFO frequency for each channel.
     * @param frequencies A span of LFO frequencies to apply to each channel.
     * The size of the span must be equal to InputChannelCount() for ParallelGainsMode::Merge.
     * The size of the span must be equal to OutputChannelCount() for ParallelGainsMode::Split.
     * The size of the span must be equal to InputChannelCount() and OutputChannelCount() for
     * ParallelGainsMode::Parallel.
     */
    void SetLfoFrequency(std::span<const float> frequencies);

    /** @brief Sets the LFO amplitude for each channel.
     * @param amplitudes A span of LFO amplitudes to apply to each channel.
     * The size of the span must be equal to InputChannelCount() for ParallelGainsMode::Merge.
     * The size of the span must be equal to OutputChannelCount() for ParallelGainsMode::Split.
     * The size of the span must be equal to InputChannelCount() and OutputChannelCount() for
     * ParallelGainsMode::Parallel.
     *
     * The amplitude is the peak deviation from the center gain.
     */
    void SetLfoAmplitude(std::span<const float> amplitudes);

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples.
     * For ParallelGainsMode::Split, the input buffer must have 1 channels and the output
     * buffer must have OutputChannelCount() channels.
     * For ParallelGainsMode::Merge, the input buffer must have InputChannelCount() channels and the output buffer must
     * have 1 channels. For ParallelGainsMode::Parallel, the input and output buffers must have the same number of
     * channels equal to InputChannelCount() and OutputChannelCount().
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Gets the number of input channels. */
    uint32_t InputChannelCount() const override;

    /** @brief Gets the number of output channels. */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the phase of all LFOs to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the TimeVaryingParallelGains processor.
     * @return A unique pointer to the cloned TimeVaryingParallelGains processor.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    void ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockParallel(const AudioBuffer& input, AudioBuffer& output);

    ParallelGainsMode mode_;
    std::vector<SineWave> lfos_;
};

} // namespace sfFDN