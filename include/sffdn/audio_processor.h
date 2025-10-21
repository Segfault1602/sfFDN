// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_buffer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace sfFDN
{
/** @defgroup AudioProcessors Audio Processors
 * @brief Classes for audio processing.
 */

/** @brief Base class for audio processors.
 * An AudioProcessor processes audio samples and outputs the result, e.g., applying effects or transformations.
 */
class AudioProcessor
{
  public:
    AudioProcessor() = default;
    virtual ~AudioProcessor() = default;

    AudioProcessor(const AudioProcessor&) = delete;
    AudioProcessor& operator=(const AudioProcessor&) = delete;
    AudioProcessor(AudioProcessor&&) = delete;
    AudioProcessor& operator=(AudioProcessor&&) = delete;

    /** @brief Process audio buffers.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     */
    virtual void Process(const AudioBuffer& input, AudioBuffer& output) noexcept = 0;

    /** @brief Returns the number of input channels this processor expects. */
    virtual uint32_t InputChannelCount() const = 0;

    /** @brief Returns the number of output channels this processor produces. */
    virtual uint32_t OutputChannelCount() const = 0;

    /** @brief Clears the internal state of the processor.
     * This function should reset any internal buffers or states used by the processor without changing its
     * configuration. For example, calling `Clear()` on an IIR filter should set its internal state to 0 while keeping
     * its coefficients intact.
     */
    virtual void Clear() = 0;

    /** @brief Creates a copy of the audio processor.
     * @return A unique pointer to the cloned audio processor.
     */
    virtual std::unique_ptr<AudioProcessor> Clone() const = 0;
};

/** @brief A chain of audio processors that processes audio sequentially.
 * This class allows adding multiple audio processors and processes the audio through each processor in the chain.
 */
class AudioProcessorChain : public AudioProcessor
{
  public:
    /** @brief Constructs an AudioProcessorChain with a specified block size.
     * @param block_size The size of the audio blocks to process.
     * @note The block size is used to allocate internal buffers for processing.
     */
    AudioProcessorChain(uint32_t block_size);

    /** @brief Adds an audio processor to the chain.
     * @param processor A unique pointer to the audio processor to add.
     * @return True if the processor was added successfully, false if the output channel count of the last processor
     * does not match the input channel count of the new processor.
     */
    bool AddProcessor(std::unique_ptr<AudioProcessor>&& processor);

    uint32_t GetProcessorCount() const;

    AudioProcessor* GetProcessor(uint32_t index) const;

    /** @brief Processes the audio buffers through the chain.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * @note The input and output buffers must have the same sample count.
     * @note The input buffer's channel count must match the first processor's input channel count,
     * and the output buffer's channel count must match the last processor's output channel count.
     * @note The sample count of the input and output buffers must match the block size specified during construction.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels this processor expects. */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels this processor produces. */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of all processors in the chain. */
    void Clear() override;

    /** @brief Creates a copy of the audio processor chain.
     * @return A unique pointer to the cloned audio processor chain.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t block_size_ = 0;
    std::vector<std::unique_ptr<AudioProcessor>> processors_;

    std::vector<float> work_buffer_a_;
    std::vector<float> work_buffer_b_;
    uint32_t max_work_buffer_size_ = 0;
};
} // namespace sfFDN