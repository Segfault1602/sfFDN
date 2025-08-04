// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "audio_processor.h"
#include "circular_buffer.h"

namespace sfFDN
{

class PartitionedConvolverSegment;

class PartitionedConvolver : public AudioProcessor
{
  public:
    PartitionedConvolver(size_t block_size, std::span<const float> fir);
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
    size_t block_size_;
    CircularBuffer output_buffer_;

    std::vector<std::unique_ptr<PartitionedConvolverSegment>> segments_;
};

} // namespace sfFDN