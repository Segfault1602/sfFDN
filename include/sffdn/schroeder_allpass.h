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

class SchroederAllpass
{
  public:
    SchroederAllpass(uint32_t delay, float g);

    void SetDelay(uint32_t delay);
    void SetG(float g);

    uint32_t GetDelay() const
    {
        return delay_.GetDelay();
    }

    float GetG() const
    {
        return g_;
    }

    float Tick(float input);
    void ProcessBlock(std::span<const float> in, std::span<float> out);

    void Clear();

  private:
    Delay delay_;
    float g_;
};

class SchroederAllpassSection : public AudioProcessor
{
  public:
    SchroederAllpassSection(uint32_t N);

    void SetDelays(std::span<uint32_t> delays);
    void SetGains(std::span<float> gains);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<SchroederAllpass> allpasses_;
};
} // namespace sfFDN