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

    /** @brief Set the coefficients of the filter.
     * @param b0 The feedforward coefficient.
     * @param a1 The feedback coefficient.
     */
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
    float Tick(float in) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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
    void SetCoefficients(float coeff) noexcept SFFDN_NONBLOCKING
    {
        coeff_ = coeff;
    }

    /** @brief Re-seeds the filter state and sets a new coefficient in one step.
     *
     * The filter is a Direct Form I first order allpass, so its state is the previous input sample and the previous
     * output sample. When the filter is used as a fractional delay and the signal feeding it is switched to a
     * different tap of a delay line, the stored previous input belongs to the old tap and is no longer the sample
     * that precedes the next input. Feeding that stale sample back into Tick() injects a step into the output; this
     * is the source of the clicks heard when an allpass interpolated delay line crosses an integer sample boundary.
     *
     * @param last_in The sample that precedes the next input, i.e. the previous output of the new tap.
     * @param coeff The new allpass coefficient.
     *
     * @note The previous output is deliberately left untouched: it is a genuine continuation of the output signal,
     * and the remaining discontinuity decays at a rate set by the coefficient.
     */
    void WarpState(float last_in, float coeff) noexcept SFFDN_NONBLOCKING
    {
        last_in_ = last_in;
        coeff_ = coeff;
    }

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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

    /** @brief Sets the biquad coefficients for each stage.
     * @param coeffs A span of FilterCoefficients, one for each biquad stage.
     */
    void SetCoefficients(std::span<const FilterCoefficients> coeffs);

    /** @brief Processes a single input sample through the filter.
     * @param in The input sample.
     * @return The output sample.
     */
    float Tick(float in) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the internal state of the processor.
     * This function resets the internal state of all biquad stages to zero.
     */
    void Clear() override;

    /** @brief Creates a copy of the filter.
     * @return A unique pointer to the cloned filter.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

    /** @brief Represents the internal state of the IIR filter.
     */
    struct IIRState
    {
        float s0, s1;
    };

  private:
    uint32_t stage_;
    std::vector<IIRState> states_;
    std::vector<FilterCoefficients> coeffs_;
};

/** @brief Implements an FIR filter with arbitrary coefficients.
 * @ingroup AudioProcessors
 */
class Fir : public AudioProcessor
{
  public:
    /** @brief Constructs a FIR filter. */
    Fir(const FirOptions& config = {});
    ~Fir() override;

    /** @brief Copy constructor for the FIR filter.
     */
    Fir(const Fir& other);

    /** @brief Copy assignment operator for the FIR filter.
     * @return A reference to the assigned FIR filter.
     */
    Fir& operator=(const Fir& other);

    /** @brief Move constructor for the FIR filter.
     */
    Fir(Fir&& other) noexcept;

    /** @brief Move assignment operator for the FIR filter.
     * @return A reference to the assigned FIR filter.
     */
    Fir& operator=(Fir&& other) noexcept;

    /** @brief Sets the FIR coefficients.
     * @param coeffs The FIR coefficients.
     */
    void SetCoefficients(std::span<const float> coeffs);

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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

/** @brief Implements a sparse FIR filter.
 * @ingroup AudioProcessors
 */
class SparseFir : public AudioProcessor
{
  public:
    /** @brief Constructs a sparse FIR filter. */
    SparseFir(const SparseFirOptions& config = {});
    ~SparseFir() override;

    // impl_ is a unique_ptr to an incomplete type, so the destructor must be defined out of line,
    // which suppresses the implicit move operations. Declare them explicitly so a SparseFir can be
    // moved instead of silently failing to compile. The pimpl cannot be copied; use Clone().
    SparseFir(const SparseFir&) = delete;
    SparseFir& operator=(const SparseFir&) = delete;
    SparseFir(SparseFir&& other) noexcept;
    SparseFir& operator=(SparseFir&& other) noexcept;

    /** @brief Sets the FIR coefficients.
     * @param config The FIR coefficients.
     */
    void SetCoefficients(const SparseFirOptions& config = {});

    /**
     * @brief Input a sample in the filter and return the next output
     * @param in The input sample
     * @return The next output sample
     */
    float Tick(float in) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a block of input samples through the filter.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output buffers must have the same number of channels and sample count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

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

    SparseFirOptions config_;
};

/** @brief Creates an Fir or SparseFir filter based on the provided configuration.
 * @param config The configuration for the FIR filter.
 * @param sparse_threshold The threshold for determining whether to create a sparse FIR filter. If the ratio of non-zero
 * coefficients to total coefficients is below this threshold, a SparseFir filter will be created. Otherwise, a regular
 * Fir filter will be created.
 * @return A unique pointer to the created FIR filter.
 */
std::unique_ptr<AudioProcessor> MakeFirFilter(const FirOptions& config, float sparse_threshold = 0.25f);

} // namespace sfFDN
