// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "audio_processor.h"

namespace sfFDN
{

enum class ParallelGainsMode : uint8_t
{
    Multiplexed,   // Process input as a single channel and output to multiple channels
    DeMultiplexed, // Process each input channel separately and output to multiple channels
    Parallel       // Process each input channel separately and output to the same number of channels
};

class ParallelGains : public AudioProcessor
{
  public:
    ParallelGains(ParallelGainsMode mode);
    ParallelGains(uint32_t N, ParallelGainsMode mode, float gain = 1.0f);
    ParallelGains(ParallelGainsMode mode, std::span<const float> gains);

    void SetMode(ParallelGainsMode mode);
    void SetGains(std::span<const float> gains);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;
    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    void ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockParallel(const AudioBuffer& input, AudioBuffer& output);

    std::vector<float> gains_;
    ParallelGainsMode mode_;
};

} // namespace sfFDN