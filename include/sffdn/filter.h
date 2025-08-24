// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "audio_processor.h"

namespace sfFDN
{

/// @brief Implements a simple one pole filter with differential equation y(n) = b0*x(n) - a1*y(n-1)
class OnePoleFilter : public AudioProcessor
{
  public:
    OnePoleFilter();

    /// @brief Set the pole of the filter.
    /// @param pole The pole of the filter.
    void SetPole(float pole);

    void SetCoefficients(float b0, float a1);

    /// @brief Set the pole of the filter to obtain an exponential decay filter.
    /// @param decayDb The decay in decibels.
    /// @param timeMs The time in milliseconds.
    /// @param samplerate The samplerate.
    void SetDecayFilter(float decayDb, float timeMs, float samplerate);

    /// @brief Set the pole of the filter to obtain a lowpass filter with a 3dB cutoff frequency.
    /// @param cutoff The cutoff frequency, normalized between 0 and 1.
    void SetLowpass(float cutoff);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float gain_;
    float b0_, a1_;
    std::array<float, 2> state_;
};

class CascadedBiquads : public AudioProcessor
{
  public:
    CascadedBiquads();

    void SetCoefficients(uint32_t num_stage, std::span<const float> coeffs);

    float Tick(float in);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

    void dump_coeffs();

    struct IIRCoeffs
    {
        float b0, b1, b2, a1, a2;
    };

    struct IIRState
    {
        float s0, s1;
    };

  private:
    uint32_t stage_;
    std::vector<IIRState> states_;
    std::vector<IIRCoeffs> coeffs_;
};
} // namespace sfFDN