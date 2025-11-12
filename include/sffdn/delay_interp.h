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
    Linear,
    Allpass,
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
                DelayInterpolationType type = DelayInterpolationType::Linear);

    /** @brief Clears all internal states of the delay line. */
    void Clear(void);

    /** @brief Gets the maximum delay-line length. */
    uint32_t GetMaximumDelay() const
    {
        return delayline_.GetMaximumDelay();
    }

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
    float GetDelay() const
    {
        return delay_;
    }

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

  private:
    Delay delayline_;
    DelayInterpolationType type_;

    float delay_;
    uint32_t int_delay_;
    float frac_delay_;

    AllpassFilter allpass_;

    void ProcessLinear(const AudioBuffer& input, AudioBuffer& output);
    void ProcessAllpass(const AudioBuffer& input, AudioBuffer& output);
};

} // namespace sfFDN