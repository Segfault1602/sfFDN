// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "sffdn/nonlinear.h"

#include "dc_blocker.h"
#include "passthrough.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/filterbank.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace
{
/** @brief The antiderivative of `|x|`. */
float AbsoluteAntiderivative(float x) noexcept SFFDN_NONBLOCKING
{
    return 0.5f * x * std::abs(x);
}

/** @brief Returns the power compensation gain of equation (3) of the paper. */
float CompensationGain(float alpha)
{
    return std::sqrt(2.f - (2.f * std::abs(alpha - 0.5f)));
}

/** @brief Throws if `alpha` is out of range, and returns it otherwise.
 *
 * Returning the value lets the constructor validate inside its member initializer list, which keeps the check ahead
 * of CompensationGain(): that computes `sqrt(2 - 2|alpha - 1/2|)`, which is NaN for an out-of-range alpha.
 */
float ValidateAlpha(float alpha)
{
    if (!std::isfinite(alpha) || alpha < 0.f || alpha > 1.f)
    {
        throw std::invalid_argument("ControllableFullWaveRectifier: alpha must be in [0, 1]");
    }
    return alpha;
}

void ValidateD(float d)
{
    if (!std::isfinite(d) || d < 0.f || d > 1.f)
    {
        throw std::invalid_argument("SignalDependentFractionalDelay: d must be in [0, 1]");
    }
}

/** @brief Returns the index of the first channel that carries a processor.
 *
 * `active_channel_count` counts back from the last channel, and zero means every channel.
 */
uint32_t FirstActiveChannel(uint32_t channel_count, uint32_t active_channel_count)
{
    if (active_channel_count == 0 || active_channel_count >= channel_count)
    {
        return 0;
    }
    return channel_count - active_channel_count;
}
} // namespace

namespace sfFDN
{

ControllableFullWaveRectifier::ControllableFullWaveRectifier(const ControllableFullWaveRectifierOptions& options)
    : alpha_(ValidateAlpha(options.alpha))
    , compensation_gain_(CompensationGain(options.alpha))
    , antialiasing_(options.antialiasing)
{
    if (options.dc_block)
    {
        if (!std::isfinite(options.sample_rate) || options.sample_rate <= 0.f)
        {
            throw std::invalid_argument(
                "ControllableFullWaveRectifier: sample_rate must be positive when dc_block is enabled");
        }
        dc_blocker_ = std::make_unique<DcBlocker>(options.sample_rate);
    }
}

ControllableFullWaveRectifier::ControllableFullWaveRectifier(const ControllableFullWaveRectifier& other)
    : dc_blocker_(other.dc_blocker_ ? std::make_unique<DcBlocker>(*other.dc_blocker_) : nullptr)
    , alpha_(other.alpha_)
    , compensation_gain_(other.compensation_gain_)
    , antialiasing_(other.antialiasing_)
    , prev_input_(other.prev_input_)
{
}

ControllableFullWaveRectifier& ControllableFullWaveRectifier::operator=(const ControllableFullWaveRectifier& other)
{
    if (this != &other)
    {
        dc_blocker_ = other.dc_blocker_ ? std::make_unique<DcBlocker>(*other.dc_blocker_) : nullptr;
        alpha_ = other.alpha_;
        compensation_gain_ = other.compensation_gain_;
        antialiasing_ = other.antialiasing_;
        prev_input_ = other.prev_input_;
    }
    return *this;
}

ControllableFullWaveRectifier::ControllableFullWaveRectifier(ControllableFullWaveRectifier&&) noexcept = default;
ControllableFullWaveRectifier& ControllableFullWaveRectifier::operator=(ControllableFullWaveRectifier&&) noexcept =
    default;
ControllableFullWaveRectifier::~ControllableFullWaveRectifier() = default;

void ControllableFullWaveRectifier::SetAlpha(float alpha)
{
    ValidateAlpha(alpha);
    alpha_ = alpha;
    compensation_gain_ = CompensationGain(alpha);
}

float ControllableFullWaveRectifier::GetAlpha() const noexcept SFFDN_NONBLOCKING
{
    return alpha_;
}

float ControllableFullWaveRectifier::GetCompensationGain() const noexcept SFFDN_NONBLOCKING
{
    return compensation_gain_;
}

float ControllableFullWaveRectifier::Rectify(float input) const noexcept SFFDN_NONBLOCKING
{
    if (!antialiasing_)
    {
        return std::abs(input);
    }

    // First-order antiderivative antialiasing, equation (4) of the paper. The quotient is the average slope of the
    // antiderivative of |x| between the two samples, which is why it approximates |x| while suppressing the aliased
    // components of the corner at the origin.
    const float denominator = input - prev_input_;
    if (std::abs(denominator) <= kAntialiasingEpsilon)
    {
        // The quotient is ill-conditioned here: both terms of the numerator have cancelled down to noise. Fall back
        // to the midpoint, as suggested by Parker et al. and used by the reference implementation.
        return std::abs(input + prev_input_) * 0.5f;
    }

    return (AbsoluteAntiderivative(input) - AbsoluteAntiderivative(prev_input_)) / denominator;
}

float ControllableFullWaveRectifier::Tick(float input) noexcept SFFDN_NONBLOCKING
{
    const float rectified = Rectify(input);
    prev_input_ = input;

    float output = compensation_gain_ * ((alpha_ * rectified) + ((1.f - alpha_) * input));

    if (dc_blocker_ != nullptr)
    {
        output = dc_blocker_->Tick(output);
    }

    return output;
}

void ControllableFullWaveRectifier::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    // Hoisted out of the loop: GetChannelSpan is out of line, and Debug does not inline it.
    const auto in_span = input.GetChannelSpan(0);
    const auto out_span = output.GetChannelSpan(0);

    const float alpha = alpha_;
    const float dry = 1.f - alpha;
    const float gain = compensation_gain_;

    if (!antialiasing_)
    {
        for (auto i = 0u; i < in_span.size(); ++i)
        {
            const float sample = in_span[i];
            out_span[i] = gain * ((alpha * std::abs(sample)) + (dry * sample));
        }

        if (!in_span.empty())
        {
            prev_input_ = in_span.back();
        }
    }
    else
    {
        float previous = prev_input_;
        for (auto i = 0u; i < in_span.size(); ++i)
        {
            const float sample = in_span[i];
            const float denominator = sample - previous;
            const float rectified =
                std::abs(denominator) <= kAntialiasingEpsilon
                    ? std::abs(sample + previous) * 0.5f
                    : (AbsoluteAntiderivative(sample) - AbsoluteAntiderivative(previous)) / denominator;
            out_span[i] = gain * ((alpha * rectified) + (dry * sample));
            previous = sample;
        }
        prev_input_ = previous;
    }

    if (dc_blocker_ != nullptr)
    {
        // Process separately so the blocker's recursive state can stay in registers across the whole block.
        dc_blocker_->Process(out_span);
    }
}

uint32_t ControllableFullWaveRectifier::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

uint32_t ControllableFullWaveRectifier::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

void ControllableFullWaveRectifier::Clear()
{
    prev_input_ = 0.f;
    if (dc_blocker_ != nullptr)
    {
        dc_blocker_->Clear();
    }
}

std::unique_ptr<AudioProcessor> ControllableFullWaveRectifier::Clone() const
{
    return std::make_unique<ControllableFullWaveRectifier>(*this);
}

SignalDependentFractionalDelay::SignalDependentFractionalDelay(const SignalDependentFractionalDelayOptions& options)
    : d_(options.d)
{
    ValidateD(options.d);
}

void SignalDependentFractionalDelay::SetD(float d)
{
    ValidateD(d);
    d_ = d;
}

float SignalDependentFractionalDelay::GetD() const noexcept SFFDN_NONBLOCKING
{
    return d_;
}

float SignalDependentFractionalDelay::Tick(float input) noexcept SFFDN_NONBLOCKING
{
    const float positive = std::max(input, 0.f);
    const float negative = std::min(input, 0.f);

    // The negative branch interpolates between the current and the previous sample, giving it a delay of 1 - d. The
    // positive branch interpolates between the previous two, giving it a delay of 1 + d.
    const float negative_branch = (d_ * negative) + ((1.f - d_) * prev_negative_);
    const float positive_branch = (d_ * prev_prev_positive_) + ((1.f - d_) * prev_positive_);

    prev_prev_positive_ = prev_positive_;
    prev_positive_ = positive;
    prev_negative_ = negative;

    return negative_branch + positive_branch;
}

void SignalDependentFractionalDelay::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    const auto in_span = input.GetChannelSpan(0);
    const auto out_span = output.GetChannelSpan(0);

    const float d = d_;
    const float one_minus_d = 1.f - d;
    float previous_positive = prev_positive_;
    float previous_previous_positive = prev_prev_positive_;
    float previous_negative = prev_negative_;

    for (auto i = 0u; i < in_span.size(); ++i)
    {
        const float sample = in_span[i];
        const float positive = std::max(sample, 0.f);
        const float negative = std::min(sample, 0.f);

        out_span[i] = (d * negative) + (one_minus_d * previous_negative) + (d * previous_previous_positive) +
                      (one_minus_d * previous_positive);

        previous_previous_positive = previous_positive;
        previous_positive = positive;
        previous_negative = negative;
    }

    prev_positive_ = previous_positive;
    prev_prev_positive_ = previous_previous_positive;
    prev_negative_ = previous_negative;
}

uint32_t SignalDependentFractionalDelay::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

uint32_t SignalDependentFractionalDelay::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

void SignalDependentFractionalDelay::Clear()
{
    prev_positive_ = 0.f;
    prev_prev_positive_ = 0.f;
    prev_negative_ = 0.f;
}

std::unique_ptr<AudioProcessor> SignalDependentFractionalDelay::Clone() const
{
    return std::make_unique<SignalDependentFractionalDelay>(*this);
}

RingModulator::RingModulator(const RingModulatorOptions& options)
{
    if (!std::isfinite(options.frequency) || options.frequency < 0.f)
    {
        throw std::invalid_argument("RingModulator: frequency must be finite and non-negative");
    }

    if (!std::isfinite(options.amplitude))
    {
        throw std::invalid_argument("RingModulator: amplitude must be finite");
    }

    if (!std::isfinite(options.initial_phase) || options.initial_phase < 0.f || options.initial_phase > 1.f)
    {
        throw std::invalid_argument("RingModulator: initial_phase must be in [0, 1]");
    }

    lfo_.SetFrequency(options.frequency);
    lfo_.SetAmplitude(options.amplitude);
    lfo_.SetPhaseOffset(options.initial_phase);
}

float RingModulator::GetFrequency() const noexcept SFFDN_NONBLOCKING
{
    return lfo_.GetFrequency();
}

float RingModulator::GetAmplitude() const noexcept SFFDN_NONBLOCKING
{
    return lfo_.GetAmplitudeNonBlocking();
}

float RingModulator::Tick(float input) noexcept SFFDN_NONBLOCKING
{
    return input * lfo_.Tick();
}

void RingModulator::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    lfo_.Multiply(input.GetChannelSpan(0), output.GetChannelSpan(0));
}

uint32_t RingModulator::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

uint32_t RingModulator::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

void RingModulator::Clear()
{
    lfo_.ResetPhase();
}

std::unique_ptr<AudioProcessor> RingModulator::Clone() const
{
    return std::make_unique<RingModulator>(*this);
}

std::unique_ptr<FilterBank> MakeMultichannelControllableFullWaveRectifier(
    const MultichannelControllableFullWaveRectifierOptions& options)
{
    auto bank = std::make_unique<FilterBank>();
    for (const auto& channel_options : options.channels)
    {
        if (channel_options.has_value())
        {
            bank->AddFilter(std::make_unique<ControllableFullWaveRectifier>(channel_options.value()));
        }
        else
        {
            bank->AddFilter(std::make_unique<PassThrough>());
        }
    }
    return bank;
}

MultichannelControllableFullWaveRectifierOptions MakeMultichannelControllableFullWaveRectifierOptions(
    float alpha, float sample_rate, uint32_t channel_count, uint32_t active_channel_count)
{
    MultichannelControllableFullWaveRectifierOptions options;
    options.channels.resize(channel_count);

    const uint32_t first_active = FirstActiveChannel(channel_count, active_channel_count);
    for (auto channel = first_active; channel < channel_count; ++channel)
    {
        options.channels[channel] = ControllableFullWaveRectifierOptions{
            .alpha = alpha, .antialiasing = true, .dc_block = true, .sample_rate = sample_rate};
    }

    return options;
}

std::unique_ptr<FilterBank> MakeMultichannelSignalDependentFractionalDelay(
    const MultichannelSignalDependentFractionalDelayOptions& options)
{
    auto bank = std::make_unique<FilterBank>();
    for (const auto& channel_options : options.channels)
    {
        if (channel_options.has_value())
        {
            bank->AddFilter(std::make_unique<SignalDependentFractionalDelay>(channel_options.value()));
        }
        else
        {
            bank->AddFilter(std::make_unique<PassThrough>());
        }
    }
    return bank;
}

MultichannelSignalDependentFractionalDelayOptions MakeMultichannelSignalDependentFractionalDelayOptions(
    float d, uint32_t channel_count, uint32_t active_channel_count)
{
    MultichannelSignalDependentFractionalDelayOptions options;
    options.channels.resize(channel_count);

    const uint32_t first_active = FirstActiveChannel(channel_count, active_channel_count);
    for (auto channel = first_active; channel < channel_count; ++channel)
    {
        options.channels[channel] = SignalDependentFractionalDelayOptions{.d = d};
    }

    return options;
}

std::unique_ptr<FilterBank> MakeMultichannelRingModulator(const MultichannelRingModulatorOptions& options)
{
    auto bank = std::make_unique<FilterBank>();
    for (const auto& channel_options : options.channels)
    {
        if (channel_options.has_value())
        {
            bank->AddFilter(std::make_unique<RingModulator>(channel_options.value()));
        }
        else
        {
            bank->AddFilter(std::make_unique<PassThrough>());
        }
    }
    return bank;
}

MultichannelRingModulatorOptions MakeMultichannelRingModulatorOptions(float frequency, float amplitude,
                                                                      uint32_t channel_count,
                                                                      uint32_t active_channel_count)
{
    MultichannelRingModulatorOptions options;
    options.channels.resize(channel_count);

    const uint32_t first_active = FirstActiveChannel(channel_count, active_channel_count);
    const auto active_count = channel_count - first_active;

    for (auto channel = first_active; channel < channel_count; ++channel)
    {
        const float phase =
            (active_count > 0) ? static_cast<float>(channel - first_active) / static_cast<float>(active_count) : 0.f;
        options.channels[channel] =
            RingModulatorOptions{.frequency = frequency, .amplitude = amplitude, .initial_phase = phase};
    }

    return options;
}

} // namespace sfFDN
