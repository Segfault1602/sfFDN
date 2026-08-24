// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "filter.h"

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
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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

    /** @brief Move constructor for the IIR filter bank.
     */
    IIRFilterBank(IIRFilterBank&&) noexcept;
    /** @brief Move assignment operator for the IIR filter bank.
     * @return A reference to the assigned IIR filter bank.
     */
    IIRFilterBank& operator=(IIRFilterBank&&) noexcept;

    ~IIRFilterBank();

    /** @brief Clears the internal state of the processor. */
    void Clear() override;

    /** @brief Sets the biquad coefficients for each stage.
     * @param coeffs A span of FilterCoefficients, one for each biquad stage.
     * @param channel_count The number of channels (filters) in the filter bank.
     * Will throw an exception if coeffs.size() is not a multiple of channel_count.
     */
    void SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count);

    /** @brief Processes a block of input samples through the filter bank.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     * The number of channels must be equal to the number of filters in the filter bank.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     * This is equal to the number of filters in the filter bank.
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Creates a copy of the filter bank.
     * @return A unique pointer to the cloned filter bank.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class IIRFilterBankImpl;
    std::unique_ptr<IIRFilterBankImpl> impl_;

    std::vector<FilterCoefficients> coeffs_;
};

} // namespace sfFDN