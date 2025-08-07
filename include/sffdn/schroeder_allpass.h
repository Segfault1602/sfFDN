// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <span>
#include <cstdint>

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

    float Tick(float input);
    void ProcessBlock(std::span<const float> in, std::span<float> out);

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

  private:
    std::vector<SchroederAllpass> allpasses_;
    uint32_t stage_;
};
} // namespace sfFDN