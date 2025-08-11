// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "audio_processor.h"

namespace sfFDN
{
/// @brief A partitioned convolution engine that can filter audio signals with an FIR filter.
class PartitionedConvolver : public AudioProcessor
{
  public:
    /// @brief Constructs a PartitionedConvolver.
    ///
    /// @param block_size The block size to use for processing.
    /// @param fir The FIR filter coefficients.
    /// The PartitionedConvolver only works if the block size stays constant during use.
    /// Process() expects the input and output buffers to have a sample count equal to the block size.
    PartitionedConvolver(uint32_t block_size, std::span<const float> fir);
    ~PartitionedConvolver();

    PartitionedConvolver(const PartitionedConvolver&) = delete;
    PartitionedConvolver& operator=(const PartitionedConvolver&) = delete;

    PartitionedConvolver(PartitionedConvolver&&) noexcept;
    PartitionedConvolver& operator=(PartitionedConvolver&&) noexcept;

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

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class PartitionedConvolverImpl;
    std::unique_ptr<PartitionedConvolverImpl> impl_;

    PartitionedConvolver() = default; // Default constructor used in Clone()
};

} // namespace sfFDN