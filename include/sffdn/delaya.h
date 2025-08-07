// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>

#include "sffdn/audio_buffer.h"

namespace sfFDN
{
/// @brief Delay line with allpass interpolation.
class DelayAllpass
{
  public:
    /// @brief Constructs a delay line with allpass interpolation.
    /// @param delay The initial delay in samples. This value must be >= 0.5.
    /// @param maxDelay The maximum delay in samples.
    DelayAllpass(float delay = 0.5, uint32_t maxDelay = 4095);

    /// @brief Clears all internal states of the delay line.
    void Clear(void);

    /// @brief Gets the maximum delay-line length.
    uint32_t GetMaximumDelay() const
    {
        return buffer_.size() - 1;
    }

    /// @brief Sets the maximum delay for the delay line.
    /// @param delay The maximum delay in samples.
    void SetMaximumDelay(uint32_t delay);

    /// @brief Sets the delay for the delay line.
    /// @param delay The delay in samples. This value must be >= 0.5 and < GetMaximumDelay().
    void SetDelay(float delay);

    /// @brief Returns the current delay in samples.
    float GetDelay() const
    {
        return delay_;
    }

    /// @brief Processes a single sample through the delay line.
    /// @param input The input sample to process.
    /// @return The processed output sample.
    float Tick(float input);

    /// @brief Processes a block of input samples.
    /// @param input The input audio buffer.
    /// @param output The output audio buffer.
    void Process(const AudioBuffer& input, AudioBuffer& output);

  protected:
    uint32_t in_point_;
    uint32_t out_point_;
    float delay_;
    float alpha_;
    float coeff_;
    float ap_input_;
    float next_output_;
    bool do_next_out_;

    std::vector<float> buffer_;
    float last_frame_;
    float gain_;

  private:
    void UpdateAlpha(float delay);
    float NextOut();
};

} // namespace sfFDN