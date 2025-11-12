// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "delay.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{

/** @brief Implements a simple one pole filter with differential equation \f$y(n) = b_0x(n) - a_1y(n-1)\f$
 * @ingroup AudioProcessors
 */
class OnePoleFilter : public AudioProcessor
{
  public:
    /** @brief Constructs a one pole filter. */
    OnePoleFilter();

    /** @brief Sets the pole of the filter based on T60 times.
     * @param dc The T60 time in seconds at DC (0 Hz).
     * @param ny The T60 time in seconds at Nyquist frequency.
     * @param delay The delay in samples for the delay line preceding the filter.
     * @param sample_rate The sample rate in Hz.
     *
     * See sfFDN::GetOnePoleAbsorption.
     */
    void SetT60s(float dc, float ny, uint32_t delay, float sample_rate);

    /**
     * @brief Set the pole of the filter.
     * @param pole The pole of the filter.
     */
    void SetPole(float pole);

    void SetCoefficients(float b0, float a1);

    /**
     * @brief Set the pole of the filter to obtain an exponential decay filter.
     * @param decay_db The decay in decibels.
     * @param time_ms The time in milliseconds.
     * @param sample_rate The sample rate.
     */
    void SetDecayFilter(float decay_db, float time_ms, float sample_rate);

    /**
     * @brief Set the pole of the filter to obtain a lowpass filter with a 3dB cutoff frequency.
     * @param cutoff The cutoff frequency, normalized between 0 and 1.
     */
    void SetLowpass(float cutoff);

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in);

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of the filter to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float gain_;
    float b0_, a1_;
    std::array<float, 2> state_;
};

/** @brief Implements a simple allpass filter with differential equation \f$y(n) = g*x(n) + x(n-1) -g*y(n-1)\f$
 * @ingroup AudioProcessors
 */
class AllpassFilter : public AudioProcessor
{
  public:
    /** @brief Constructs an allpass filter. */
    AllpassFilter();

    AllpassFilter(const AllpassFilter&);
    AllpassFilter& operator=(const AllpassFilter&);

    AllpassFilter(AllpassFilter&&) noexcept;
    AllpassFilter& operator=(AllpassFilter&&) noexcept;

    /** @brief Sets the allpass coefficient.
     * @param coeff The allpass coefficient.
     */
    void SetCoefficients(float coeff);

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in);

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of the filter to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float coeff_;
    float last_in_;
    float last_out_;
};

/** @brief Implements a cascade of biquad IIR filters.
 * @ingroup AudioProcessors
 */
class CascadedBiquads : public AudioProcessor
{
  public:
    /** @brief Constructs a cascaded biquad filter. */
    CascadedBiquads();
    ~CascadedBiquads() = default;

    CascadedBiquads(const CascadedBiquads&);
    CascadedBiquads& operator=(const CascadedBiquads&);

    CascadedBiquads(CascadedBiquads&&) noexcept;
    CascadedBiquads& operator=(CascadedBiquads&&) noexcept;

    /** @brief Sets the number of biquad stages in the cascade.
     * @param num_stage The number of biquad stages.
     * @param coeffs The biquad coefficients in the format.
     * If coeffs.size() == num_stage * 5, the coefficients are assumed to be in the format
     * {b0, b1, b2, a1, a2} for each stage.
     * If coeffs.size() == num_stage * 6, the coefficients are assumed to be in the format
     * {b0, b1, b2, a0, a1, a2} for each stage.
     */
    void SetCoefficients(uint32_t num_stage, std::span<const float> coeffs);

    /** @brief Processes a single input sample through the filter.
     * @param in The input sample.
     * @return The output sample.
     */
    float Tick(float in);

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of all biquad stages to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

    struct IIRCoeffs
    {
        float b0, b1, b2, a1, a2;
    };

    struct IIRState
    {
        float s0, s1;
    };

  private:
    uint32_t stage_;
    std::vector<IIRState> states_;
    std::vector<IIRCoeffs> coeffs_;
};

class Fir : public AudioProcessor
{
  public:
    /** @brief Constructs a FIR filter. */
    Fir() = default;

    /** @brief Sets the FIR coefficients.
     * @param coeffs The FIR coefficients.
     */
    void SetCoefficients(std::span<const float> coeffs);

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in);

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of the filter to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<float> coeffs_;
    std::vector<float> delay_line_;
    uint32_t delay_index_;
};

class SparseFir : public AudioProcessor
{
  public:
    /** @brief Constructs a sparse FIR filter. */
    SparseFir() = default;

    /** @brief Sets the FIR coefficients.
     * @param coeffs The FIR coefficients.
     */
    void SetCoefficients(std::span<const float> coeffs, std::span<const uint32_t> indices);

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in);

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of the filter to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    std::vector<float> coeffs_;
    Delay delay_line_;

    std::vector<uint32_t> sparse_index_;
    uint32_t filter_order_;
};

} // namespace sfFDN
