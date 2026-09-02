// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"

#include <algorithm>
#include <cmath>
#include <span>

namespace sfFDN
{

/** @brief A first-order dc blocker with slow adaptive energy compensation.
 *
 * Implements equations (7) and (8) of G. Dal Santo, X. Pi, K. Prawda, S. J. Schlecht and V. Välimäki, "Shimmer
 * Reverberation with Nonlinear Feedback Delay Networks", DAFx26:
 *
 * \f[
 * y(n) = g_\mathrm{env}(n) \left( x(n) - x(n-1) + R\,y(n-1) \right).
 * \f]
 *
 * The nonlinearities of the paper produce a dc component that would otherwise accumulate inside the feedback loop of
 * an FDN and drive it unstable. Removing it costs energy, so the blocker is followed by a slowly varying gain that
 * tracks the ratio of its input and output power and restores the level.
 *
 * Three details follow the reference implementation rather than the printed equations, because the paper's form is
 * either ill-conditioned or under-specified. See `.github/notes/shimmer-nonlinearities.md`.
 *
 * - The input and output *powers* are smoothed first and the square root is taken afterwards. Equation (8) reads as
 *   though the instantaneous ratio \f$\sqrt{x^2/y^2}\f$ were smoothed, which is unbounded whenever the output
 *   crosses zero.
 * - A second, faster smoother is applied to the resulting gain. The paper only mentions the 50 ms envelope.
 * - The compensation gain is applied outside the difference equation: `y(n-1)` in the recursion is the uncompensated
 *   output. Feeding the compensated value back would make the pole of the blocker move with the gain.
 *
 * Internal: not part of the public API. `ControllableFullWaveRectifier` owns one of these.
 */
class DcBlocker
{
  public:
    /** @brief Pole radius of the blocker. Corresponds to a cutoff of about 76 Hz at 96 kHz, as used in the paper. */
    static constexpr float kPoleRadius = 0.995f;

    /** @brief Time constant, in seconds, of the input and output power envelopes. */
    static constexpr float kEnvelopeTau = 0.05f;

    /** @brief Time constant, in seconds, of the smoother applied to the compensation gain. */
    static constexpr float kGainTau = 0.02f;

    /** @brief Ceiling on the compensation gain.
     *
     * The blocker sits inside a feedback loop, so an unbounded make-up gain is a way to make the network diverge on
     * near-silent input, where the output power is dominated by the dc that was just removed. The reference FDN
     * implementation caps the gain at this value for the same reason.
     */
    static constexpr float kMaxGain = 4.f;

    /** @brief Floor added to both smoothed powers before dividing them. */
    static constexpr float kPowerFloor = 1e-12f;

    /** @brief Constructs a dc blocker.
     * @param sample_rate The sample rate in Hz, used to convert the two time constants into smoothing coefficients.
     * @note The coefficients are computed here so that Process() only has to do arithmetic.
     */
    explicit DcBlocker(float sample_rate)
        : envelope_coeff_(std::exp(-1.f / (sample_rate * kEnvelopeTau)))
        , gain_coeff_(std::exp(-1.f / (sample_rate * kGainTau)))
    {
    }

    /** @brief Processes one sample. */
    float Tick(float input) noexcept SFFDN_NONBLOCKING
    {
        State state{.prev_input = prev_input_,
                    .prev_output = prev_output_,
                    .input_power = input_power_,
                    .output_power = output_power_,
                    .gain = gain_};
        const float output = TickImpl(input, state, envelope_coeff_, gain_coeff_);
        StoreState(state);
        return output;
    }

    /** @brief Processes a block in place while carrying the recursive state in registers. */
    void Process(std::span<float> buffer) noexcept SFFDN_NONBLOCKING
    {
        // Keeping the recurrences local avoids member reloads after each buffer store, which may alias this object.
        State state{.prev_input = prev_input_,
                    .prev_output = prev_output_,
                    .input_power = input_power_,
                    .output_power = output_power_,
                    .gain = gain_};
        const float envelope_coeff = envelope_coeff_;
        const float gain_coeff = gain_coeff_;

        for (float& sample : buffer)
        {
            sample = TickImpl(sample, state, envelope_coeff, gain_coeff);
        }

        StoreState(state);
    }

    /** @brief Resets the filter and envelope state. The smoothing coefficients are left untouched. */
    void Clear() noexcept SFFDN_NONBLOCKING
    {
        prev_input_ = 0.f;
        prev_output_ = 0.f;
        input_power_ = kPowerFloor;
        output_power_ = kPowerFloor;
        gain_ = 1.f;
    }

  private:
    struct State
    {
        float prev_input;
        float prev_output;
        float input_power;
        float output_power;
        float gain;
    };

    static float TickImpl(float input, State& state, float envelope_coeff, float gain_coeff) noexcept SFFDN_NONBLOCKING
    {
        const float blocked = input - state.prev_input + (kPoleRadius * state.prev_output);
        state.prev_input = input;
        state.prev_output = blocked;

        state.input_power = (envelope_coeff * state.input_power) + ((1.f - envelope_coeff) * input * input);
        state.output_power = (envelope_coeff * state.output_power) + ((1.f - envelope_coeff) * blocked * blocked);

        const float target =
            std::min(std::sqrt((state.input_power + kPowerFloor) / (state.output_power + kPowerFloor)), kMaxGain);
        state.gain = (gain_coeff * state.gain) + ((1.f - gain_coeff) * target);

        return state.gain * blocked;
    }

    void StoreState(const State& state) noexcept SFFDN_NONBLOCKING
    {
        prev_input_ = state.prev_input;
        prev_output_ = state.prev_output;
        input_power_ = state.input_power;
        output_power_ = state.output_power;
        gain_ = state.gain;
    }

    float envelope_coeff_;
    float gain_coeff_;

    float prev_input_{0.f};
    float prev_output_{0.f};

    float input_power_{kPowerFloor};
    float output_power_{kPowerFloor};
    float gain_{1.f};
};

} // namespace sfFDN
