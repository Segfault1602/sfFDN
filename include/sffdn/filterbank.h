// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
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

} // namespace sfFDN