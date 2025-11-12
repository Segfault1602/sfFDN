// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delay_time_varying.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{

/**
 * @brief A bank of parallel delay lines, each with its own delay setting. Used for processing multiple channels
 * of audio with different delays.
 * @note The delay lines are instances of the Delay class, which is a non-interpolating delay line.
 *
 * @ingroup AudioProcessors
 */
class DelayBankTimeVarying : public AudioProcessor
{
  public:
    /**
     * @brief Constructs a delay bank with a specified set of delays and maximum delay.
     * @param delays A span of delay values for each channel.
     * @param max_delay The maximum delay in samples.
     */
    DelayBankTimeVarying(std::span<const float> delays, uint32_t max_delay, DelayInterpolationType type);

    ~DelayBankTimeVarying() = default;

    DelayBankTimeVarying(const DelayBankTimeVarying&);
    DelayBankTimeVarying& operator=(const DelayBankTimeVarying&);

    DelayBankTimeVarying(DelayBankTimeVarying&&) noexcept;
    DelayBankTimeVarying& operator=(DelayBankTimeVarying&&) noexcept;

    /**
     * @brief Sets the maximum delay for all delay lines in the bank.
     * @param delay The maximum delay in samples.
     * @param block_size The size of the audio blocks to be processed in the main loop.
     * @note This can increase the size of the internal buffers if the new maximum delay is larger than the current
     * buffer size.
     * @note block_size is used to determine the optimal size of the internal buffers for each delay line.
     */
    void SetDelays(const std::span<const uint32_t> delays, uint32_t block_size = 512);

    void SetMods(const std::span<const float> freqs, const std::span<const float> depths,
                 const std::span<const float> phase_offsets = {});

    /**
     * @brief Returns the current delays for each delay line in the bank.
     * @return A vector of delay values for each channel.
     */
    std::vector<uint32_t> GetDelays() const;

    /**
     * @brief Returns the number of input channels this processor expects.
     * @return The number of input channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t InputChannelCount() const override;

    /**
     * @brief Returns the number of output channels this processor produces.
     * @return The number of output channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t OutputChannelCount() const override;

    /**
     * @brief Clears the internal delay buffers.
     * This sets all delay buffers to zero.
     */
    void Clear() override;

    /**
     * @brief Processes a block of multi-channel audio.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * @note The input and output buffers must have the same sample count and a channel count equal to the number of
     * delay lines in the bank.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Creates a copy of the delay bank.
     * @return A unique pointer to the cloned delay bank.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<DelayTimeVarying> delays_;
};
} // namespace sfFDN