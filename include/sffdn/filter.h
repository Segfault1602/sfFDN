// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"
#include "sffdn/types.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
    OnePoleFilter(float b0 = 1.f, float a1 = 0.f);

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
    AllpassFilter(const AllpassFilterOptions& config = {});

    /** @brief Sets the allpass coefficient.
     * @param coeff The allpass coefficient.
     */
    void SetCoefficients(float coeff)
    {
        coeff_ = coeff;
    }

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
    CascadedBiquads(const CascadedBiquadsOptions& config = {});
    ~CascadedBiquads() = default;

    /** @brief Sets the biquad coefficients for each stage.
     * @param coeffs A span of FilterCoefficients, one for each biquad stage.
     */
    void SetCoefficients(std::span<const FilterCoefficients> coeffs);

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

    struct IIRState
    {
        float s0, s1;
    };

  private:
    uint32_t stage_;
    std::vector<IIRState> states_;
    std::vector<FilterCoefficients> coeffs_;
};

class Fir : public AudioProcessor
{
  public:
    /** @brief Constructs a FIR filter. */
    Fir(const FirOptions& config = {});
    ~Fir();

    Fir(const Fir&);
    Fir& operator=(const Fir&);

    Fir(Fir&&) noexcept;
    Fir& operator=(Fir&&) noexcept;

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
    class FirImpl;
    std::unique_ptr<FirImpl> impl_;
};

class SparseFir : public AudioProcessor
{
  public:
    /** @brief Constructs a sparse FIR filter. */
    SparseFir(const SparseFirOptions& config = {});
    ~SparseFir();

    /** @brief Sets the FIR coefficients.
     * @param coeffs The FIR coefficients.
     */
    void SetCoefficients(const SparseFirOptions& config = {});

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
    class SparseFirImpl;
    std::unique_ptr<SparseFirImpl> impl_;
};

std::unique_ptr<AudioProcessor> MakeFirFilter(const FirOptions& config, float sparse_threshold = 0.25f);

} // namespace sfFDN
