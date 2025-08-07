// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <cstdint>

#include "audio_processor.h"

namespace sfFDN
{
class PartitionedConvolver : public AudioProcessor
{
  public:
    PartitionedConvolver(uint32_t block_size, std::span<const float> fir);
    ~PartitionedConvolver();

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void DumpInfo() const;

    uint32_t InputChannelCount() const override
    {
        return 1; // PartitionedConvolver processes one channel at a time
    }
    uint32_t OutputChannelCount() const override
    {
        return 1; // PartitionedConvolver processes one channel at a time
    }

  private:
    class PartitionedConvolverImpl;
    std::unique_ptr<PartitionedConvolverImpl> impl_;
};

} // namespace sfFDN