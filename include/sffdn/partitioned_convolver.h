// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

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
     * @param rep_count The number of times to repeat each block in the partitioned convolution. A value of zero
     * automatically selects a schedule based on the FIR length.
     * The PartitionedConvolver only works if the block size stays constant during use.
     * Process() expects the input and output buffers to have a sample count equal to the block size.
     */
    PartitionedConvolver(uint32_t block_size, std::span<const float> fir, uint32_t rep_count = 0);
    ~PartitionedConvolver() override;

    PartitionedConvolver(const PartitionedConvolver&) = delete;
    PartitionedConvolver& operator=(const PartitionedConvolver&) = delete;

    /** @brief Move constructor for the partitioned convolver.*/
    PartitionedConvolver(PartitionedConvolver&& other) noexcept;

    /** @brief Move assignment operator for the partitioned convolver.
     * @return A reference to the assigned partitioned convolver.
     */
    PartitionedConvolver& operator=(PartitionedConvolver&& other) noexcept;

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of samples equal to the block size.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the block size used for processing.
     * @returns The block size used for processing.
     */
    uint32_t GetBlockSize() const;

    /** @brief Dumps internal information to the standard output for debugging purposes.
     */
    void DumpInfo() const;

    /** @brief Gets a short string representation of the internal state of the convolver for debugging purposes.
     * @returns A short string representation of the internal state of the convolver.
     */
    std::string GetShortInfo() const;

    /** @brief Gets the number of input channels supported.
     * This is always 1, as PartitionedConvolver processes one channel at a time.
     * @returns The number of input channels supported.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return 1; // PartitionedConvolver processes one channel at a time
    }

    /** @brief Gets the number of output channels supported.
     * This is always 1, as PartitionedConvolver processes one channel at a time.
     * @returns The number of output channels supported.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
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