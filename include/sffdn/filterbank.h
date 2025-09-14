// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "audio_processor.h"

namespace sfFDN
{
class FilterBank : public AudioProcessor
{
  public:
    FilterBank();

    void Clear() override;

    void AddFilter(std::unique_ptr<AudioProcessor> filter);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<std::unique_ptr<AudioProcessor>> filters_;
};

class IIRFilterBank : public AudioProcessor
{
  public:
    IIRFilterBank();

    IIRFilterBank(const IIRFilterBank&) = delete;
    IIRFilterBank& operator=(const IIRFilterBank&) = delete;

    IIRFilterBank(IIRFilterBank&&) noexcept;
    IIRFilterBank& operator=(IIRFilterBank&&) noexcept;

    ~IIRFilterBank();

    void Clear() override;

    void SetFilter(std::span<float> coeffs, uint32_t channel_count, uint32_t stage_count);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class IIRFilterBankImpl;
    std::unique_ptr<IIRFilterBankImpl> impl_;
};

} // namespace sfFDN