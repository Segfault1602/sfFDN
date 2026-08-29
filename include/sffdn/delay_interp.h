// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"
#include "sffdn/filter.h"
#include "sffdn/types.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sfFDN
{

/** @brief Delay line with interpolation.
 *
 * @ingroup AudioProcessors
 */
class DelayInterp : public AudioProcessor
{
  public:
    /**
     * @brief Constructs a delay line with interpolation.
     * @param config The configuration for the delay line.
     */
    DelayInterp(const DelayOptions& config = {});

    /** @brief Clears all internal states of the delay line. */
    void Clear() override;

    /** @brief Gets the maximum delay-line length. */
    uint32_t GetMaximumDelay() const;

    /**
     * @brief Sets the maximum delay for the delay line.
     * @param delay The maximum delay in samples.
     */
    void SetMaximumDelay(uint32_t delay);

    /**
     * @brief Sets the delay for the delay line.
     * @param delay The delay in samples.
     * @note The smallest delay that can be realised depends on the interpolation type: 0.5 samples for Allpass and
     * 1 sample for Lagrange. Smaller values are clamped.
     */
    void SetDelay(float delay) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the current delay in samples. */
    float GetDelay() const;

    /**
     * @brief Processes a single sample through the delay line.
     * @param input The input sample to process.
     * @return The processed output sample.
     */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Returns the next output sample without writing a new input sample.
     * @return The interpolated output sample for the current delay.
     * @note This is the read-before-write counterpart of Tick(). Calling NextOut() followed by
     * Advance(input) returns the same output sample as Tick(input), but allows the output sample to be used to
     * compute the input sample, which is required to close a feedback loop around the delay line.
     * @note The current delay must be at least one sample. For the Lagrange interpolation type, the current delay must
     * be at least two samples.
     * @note The returned value is cached until the next call to Advance(), Tick(), SetDelay() or Clear(), so calling
     * NextOut() several times in a row is safe and returns the same value.
     */
    float NextOut() noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Writes the next input sample into the delay line and advances it.
     * @param input The input sample to write.
     * @note This is the read-before-write counterpart of Tick(). See NextOut().
     */
    void Advance(float input) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Taps the delay line at a fixed integer delay, without interpolation.
     * @param tap The tap point in samples.
     * @return The sample stored at the tap point.
     * @note The tap is counted from the write pointer, so TapOut(0) returns the most recently written sample. After
     * Advance(x[n]), TapOut(tap) returns x[n - tap]; before the write, it returns x[n - 1 - tap].
     * @note The tap is independent of the current delay and of the interpolation type.
     */
    float TapOut(uint32_t tap) const noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Processes a block of input samples.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /**
     * @brief Adds the next input samples to the delay line.
     * @param input The input samples to add.
     * @return True if the samples were added successfully, false otherwise.
     * @note A return value of false indicates that there was not enough space in the internal buffer to write the
     * input samples. In this case, the internal state remains unchanged. When processing audio in blocks, the delay
     * line maximum delay should be set to a value that is larger than the block size.
     */
    bool AddNextInputs(std::span<const float> input) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Gets the next output samples from the delay line.
     * @param output The output samples to fill.
     * @note This is the read-before-write counterpart of Process(): the block is read out first and written in
     * afterwards with AddNextInputs(). The current delay must therefore be at least as large as the block size, and
     * at least one sample larger than that for the Lagrange interpolation type, whose kernel straddles the
     * fractional delay.
     */
    void GetNextOutputs(std::span<float> output) noexcept SFFDN_NONBLOCKING;

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return 1;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return 1;
    }

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    static constexpr std::size_t kLagrangeOrder = 3;
    static constexpr std::size_t kLagrangeTapCount = kLagrangeOrder + 1;

    Delay delayline_;

    float delay_;
    uint32_t int_delay_;
    float frac_delay_;
    DelayInterpolationType type_;

    AllpassFilter allpass_;

    std::array<float, kLagrangeTapCount> lagrange_coeffs_{};
    float linear_last_out_;

    float next_out_ = 0.f;
    bool has_next_out_ = false;
};

} // namespace sfFDN