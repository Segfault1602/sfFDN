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
    SchroederAllpass() = default;
    SchroederAllpass(uint32_t delay, float g);

    SchroederAllpass(const SchroederAllpass&) = delete;
    SchroederAllpass& operator=(const SchroederAllpass&) = delete;

    SchroederAllpass(SchroederAllpass&&) = default;
    SchroederAllpass& operator=(SchroederAllpass&&) = default;

    ~SchroederAllpass() = default;

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
    float g_{};
};

class SchroederAllpassSection : public AudioProcessor
{
  public:
    SchroederAllpassSection() = default;
    SchroederAllpassSection(uint32_t filter_count);

    SchroederAllpassSection(const SchroederAllpassSection&) = delete;
    SchroederAllpassSection& operator=(const SchroederAllpassSection&) = delete;

    SchroederAllpassSection(SchroederAllpassSection&&) noexcept;
    SchroederAllpassSection& operator=(SchroederAllpassSection&&) noexcept;

    ~SchroederAllpassSection() = default;

    void SetFilterCount(uint32_t filter_count);

    void SetDelays(std::span<const uint32_t> delays);
    void SetGains(std::span<const float> gains);
    void SetGain(float gain);

    std::vector<uint32_t> GetDelays() const;
    std::vector<float> GetGains() const;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<SchroederAllpass> allpasses_;
};

/// \brief A section of parallel Schroeder allpass filters.
class ParallelSchroederAllpassSection : public AudioProcessor
{
  public:
    ParallelSchroederAllpassSection(uint32_t N, uint32_t order);

    void SetDelays(std::span<const uint32_t> delays);
    void SetGains(std::span<const float> gains);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<SchroederAllpassSection> allpasses_;
    uint32_t order_;
};
} // namespace sfFDN