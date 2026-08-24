// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "attributes.h"
#include "audio_buffer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{

/** @brief A simple non-interpolating delay line implementation. */
class Delay
{
  public:
    /**
     * @brief Constructs a delay line with a specified delay and maximum delay.
     * @param delay The initial delay in samples.
     * @param max_delay The maximum delay in samples, ie. the size of the internal buffer.
     */
    Delay(uint32_t delay = 0, uint32_t max_delay = 4095);

    /** @brief Clears the delay line, resetting the internal buffer. The delay value remains unchanged. */
    void Clear();

    /**
     * @brief Sets the maximum delay for the delay line.
     * @param delay The maximum delay in samples.
     * @note This can increase the size of the internal buffer if the new maximum delay is larger than the current
     * buffer size.
     */
    void SetMaximumDelay(uint32_t delay);

    /**
     * @brief Gets the maximum delay-line length.
     * @return The maximum delay in samples.
     */
    uint32_t GetMaximumDelay() const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Sets the delay for the delay line.
     * @param delay The delay in samples.
     */
    void SetDelay(uint32_t delay) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the current delay in samples. */
    uint32_t GetDelay(void) const noexcept SFFDN_NONBLOCKING
    {
        return delay_;
    };

    /**
     * @brief Returns the last output sample from the delay line.
     * @return The last output sample.
     */
    float LastOut() const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Returns the next output sample.
     * @return The next output sample.
     */
    float NextOut() const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Processes the next input sample.
     * @param input The input sample to process.
     * @return The processed output sample.
     */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Taps the output of the delay line at a specific point.
     * @param tap The tap point in samples.
     * @return The output sample at the tap point.
     */
    float TapOut(uint32_t tap) const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Processes a block of input samples.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * @note The input and output buffers must have the same sample count and a channel count of 1 (mono).
     * @note The input samples are added to the delay line, and the output samples are read from the delay line.
     */
    void Process(const AudioBuffer input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Adds the next input samples to the delay line.
     * @param input The input samples to add.
     * @return True if the samples were added successfully, false otherwise.
     * @note A return value of false indicates that there was not enough space in the internal buffer to write the
     * input samples. In this case, the internal state remains unchanged. When processing audio in blocks, the delay
     * line maximum delay should be set to a value that is larger than the block size.
     */
    bool AddNextInputs(std::span<const float> input) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns whether the complete input block can be added without overwriting unread samples. */
    bool CanAddNextInputs(size_t sample_count) const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Gets the next output samples from the delay line.
     * @param output The output samples to fill.
     */
    void GetNextOutputs(std::span<float> output) noexcept SFFDN_NONBLOCKING;

    /** @brief Gets the next output samples from the delay line at specific tap points.
     * @param taps The tap points in samples.
     * @param output The output samples to fill.
     * @param coeffs The coefficients to apply to each tap point when filling the output buffer. Must have the same size
     * as `taps`.
     */
    void GetNextOutputsAt(std::span<uint32_t> taps, std::span<float> output,
                          std::span<float> coeffs) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Get a span pointing to the next writable region of the buffer
     *
     * @param size The desired size of the buffer.
     * @return std::span<float>
     *
     * Need to call AdvanceRead after reading from the buffer to update the internal state of the delay line.
     */
    std::span<float> GetNextOutputBuffers(uint32_t size) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Get a span pointing to the next writable region of the buffer
     *
     * @param size The desired size of the buffer.
     * @return std::span<float>
     *
     * Need to call AdvanceWrite after writing to the buffer to update the internal state of the delay line.
     */
    std::span<float> GetNextInputBuffers(uint32_t size) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Gets the next read and write buffers for the delay line.
     * @param read_buffer The span to fill with the next readable region of the buffer.
     * @param write_buffer The span to fill with the next writable region of the buffer.
     * @param size The desired size of the buffers.
     */
    void GetNextReadAndWriteBuffers(std::span<float>& read_buffer, std::span<float>& write_buffer,
                                    uint32_t size) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Advances the write pointer of the delay line by a specified number of samples.
     *
     * @param sample_count The number of samples to advance the write pointer.
     */
    void AdvanceWrite(uint32_t sample_count) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Advances the read pointer of the delay line by a specified number of samples.
     *
     * @param sample_count The number of samples to advance the read pointer.
     */
    void AdvanceRead(uint32_t sample_count) noexcept SFFDN_NONBLOCKING;

  private:
    uint32_t in_point_;
    uint32_t out_point_;
    uint32_t delay_;
    std::vector<float> buffer_;
    float last_frame_;
};

} // namespace sfFDN