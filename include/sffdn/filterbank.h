// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sfFDN
{
/** @brief Implements a bank of filters.
 * @ingroup AudioProcessors
 */
class FilterBank : public AudioProcessor
{
  public:
    /** @brief Constructs an empty filter bank. */
    FilterBank();

    /** @brief Clears the filter bank. */
    void Clear() override;

    /** @brief Adds a filter to the filter bank.
     * @param filter A unique pointer to the filter to add.
     * The FilterBank takes ownership of the filter.
     */
    void AddFilter(std::unique_ptr<AudioProcessor> filter);

    /** @brief Processes a block of input samples through the filter bank.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     * The number of channels must be equal to the number of filters in the filter bank.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Creates a copy of the filter bank.
     * @return A unique pointer to the cloned filter bank.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<std::unique_ptr<AudioProcessor>> filters_;
};

/** @brief Implements a bank of IIR filters.
 * On MacOS, this uses the Accelerate framework for optimized processing.
 * On other platforms, this is equivalent to using a FilterBank with CascadedBiquads filters.
 * @ingroup AudioProcessors
 */
class IIRFilterBank : public AudioProcessor
{
  public:
    /** @brief Constructs an empty IIR filter bank. */
    IIRFilterBank();

    IIRFilterBank(const IIRFilterBank&) = delete;
    IIRFilterBank& operator=(const IIRFilterBank&) = delete;

    IIRFilterBank(IIRFilterBank&&) noexcept;
    IIRFilterBank& operator=(IIRFilterBank&&) noexcept;

    ~IIRFilterBank();

    /** @brief Clears the internal state of the processor. */
    void Clear() override;

    /** @brief Sets the filter coefficients for the filter bank.
     * @param coeffs The filter coefficients in the format.
     * If coeffs.size() == channel_count * stage_count * 5, the coefficients are assumed to be in the format
     * {b0, b1, b2, a1, a2} for each stage.
     * If coeffs.size() == channel_count * stage_count * 6, the coefficients are assumed to be in the format
     * {b0, b1, b2, a0, a1, a2} for each stage.
     * @param channel_count The number of channels (filters) in the filter bank.
     * @param stage_count The number of biquad stages per filter.
     */
    void SetFilter(std::span<float> coeffs, uint32_t channel_count, uint32_t stage_count);

    /** @brief Processes a block of input samples through the filter bank.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     * The number of channels must be equal to the number of filters in the filter bank.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Creates a copy of the filter bank.
     * @return A unique pointer to the cloned filter bank.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class IIRFilterBankImpl;
    std::unique_ptr<IIRFilterBankImpl> impl_;
};

} // namespace sfFDN