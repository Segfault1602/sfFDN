// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "audio_processor.h"

namespace sfFDN
{
/** @brief A partitioned convolution engine that can filter audio signals with an FIR filter.
 * @ingroup AudioProcessors
 */
class PartitionedConvolver : public AudioProcessor
{
  public:
    /**
     * @brief Constructs a PartitionedConvolver.
     *
     * @param block_size The block size to use for processing.
     * @param fir The FIR filter coefficients.
     * The PartitionedConvolver only works if the block size stays constant during use.
     * Process() expects the input and output buffers to have a sample count equal to the block size.
     */
    PartitionedConvolver(uint32_t block_size, std::span<const float> fir);
    ~PartitionedConvolver();

    PartitionedConvolver(const PartitionedConvolver&) = delete;
    PartitionedConvolver& operator=(const PartitionedConvolver&) = delete;

    PartitionedConvolver(PartitionedConvolver&&) noexcept;
    PartitionedConvolver& operator=(PartitionedConvolver&&) noexcept;

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples equal to the block size.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Dumps internal information to the standard output for debugging purposes.
     */
    void DumpInfo() const;

    /** @brief Gets the number of input channels supported.
     * This is always 1, as PartitionedConvolver processes one channel at a time.
     * @returns The number of input channels supported.
     */
    uint32_t InputChannelCount() const override
    {
        return 1; // PartitionedConvolver processes one channel at a time
    }

    /** @brief Gets the number of output channels supported.
     * This is always 1, as PartitionedConvolver processes one channel at a time.
     * @returns The number of output channels supported.
     */
    uint32_t OutputChannelCount() const override
    {
        return 1; // PartitionedConvolver processes one channel at a time
    }

    /** @brief Clears the internal state of the processor.
     * This function resets the internal buffers and states of the convolver, but keeps the FIR filter intact.
     */
    void Clear() override;

    /** @brief Creates a copy of the PartitionedConvolver.
     * @return A unique pointer to the cloned PartitionedConvolver.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class PartitionedConvolverImpl;
    std::unique_ptr<PartitionedConvolverImpl> impl_;

    PartitionedConvolver() = default; // Default constructor used in Clone()
};

} // namespace sfFDN