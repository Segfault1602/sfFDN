// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/delay.h"
#include "sffdn/filter.h"

#include <cstdint>
#include <vector>

namespace sfFDN
{

enum class DelayInterpolationType : uint8_t
{
    None,
    Linear,
    Allpass,
    Lagrange,
};

/** @brief Delay line with interpolation. */
class DelayInterp
{
  public:
    /**
     * @brief Constructs a delay line with interpolation.
     * @param delay The initial delay in samples.
     * @param max_delay The maximum delay in samples.
     */
    DelayInterp(float delay = 0.5, uint32_t max_delay = 4095,
                DelayInterpolationType type = DelayInterpolationType::None);

    /** @brief Clears all internal states of the delay line. */
    void Clear();

    /** @brief Gets the maximum delay-line length. */
    uint32_t GetMaximumDelay() const;

    /**
     * @brief Sets the maximum delay for the delay line.
     * @param delay The maximum delay in samples.
     */
    void SetMaximumDelay(uint32_t delay);

    /**
     * @brief Sets the delay for the delay line.
     * @param delay The delay in samples.
     */
    void SetDelay(float delay);

    /** @brief Returns the current delay in samples. */
    float GetDelay() const;

    /**
     * @brief Processes a single sample through the delay line.
     * @param input The input sample to process.
     * @return The processed output sample.
     */
    float Tick(float input);

    /**
     * @brief Processes a block of input samples.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output);

    /**
     * @brief Adds the next input samples to the delay line.
     * @param input The input samples to add.
     * @return True if the samples were added successfully, false otherwise.
     * @note A return value of false indicates that there was not enough space in the internal buffer to write the
     * input samples. In this case, the internal state remains unchanged. When processing audio in blocks, the delay
     * line maximum delay should be set to a value that is larger than the block size.
     */
    bool AddNextInputs(std::span<const float> input);

    /**
     * @brief Gets the next output samples from the delay line.
     * @param output The output samples to fill.
     */
    void GetNextOutputs(std::span<float> output);

    nlohmann::json ToJson() const;

    static DelayInterp FromJson(const nlohmann::json& j);

  private:
    Delay delayline_;

    float delay_;
    uint32_t int_delay_;
    float frac_delay_;
    DelayInterpolationType type_;

    AllpassFilter allpass_;

    std::vector<float> lagrange_coeffs_;
    Fir lagrange_filter_;
};

} // namespace sfFDN