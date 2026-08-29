// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "sffdn/dattorro_delay.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/delay_interp.h"
#include "sffdn/filterbank.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
constexpr float kMaxFeedback = 0.999f;
constexpr uint32_t kDelayHeadroom = 64;

// Peak relative deviation applied to the nominal delay and the modulation rate of the outermost channels of a
// multichannel bank. Small enough that every channel still sounds like the preset, large enough that the channels do
// not drift back into lockstep.
constexpr float kChannelSpread = 0.1f;

uint32_t RequiredMaximumDelay(float delay, float width)
{
    return static_cast<uint32_t>(std::ceil(delay + std::abs(width))) + kDelayHeadroom;
}

sfFDN::DattorroDelayOptions SanitizeOptions(const sfFDN::DattorroDelayOptions& options)
{
    sfFDN::DattorroDelayOptions sanitized = options;

    const float width = options.delay_config.lfo_config.has_value() ? options.delay_config.lfo_config->amplitude : 0.f;
    sanitized.delay_config.max_delay =
        std::max(options.delay_config.max_delay, RequiredMaximumDelay(options.delay_config.delay, width));

    return sanitized;
}
} // namespace

namespace sfFDN
{

DattorroDelay::DattorroDelay(const DattorroDelayOptions& options)
    : delay_(SanitizeOptions(options).delay_config)
    , base_delay_(options.delay_config.delay)
    , feedback_tap_(static_cast<uint32_t>(std::lround(options.delay_config.delay)))
    , blend_(options.blend)
    , feedforward_(options.feedforward)
    , feedback_(std::clamp(options.feedback, -kMaxFeedback, kMaxFeedback))
{
    if (base_delay_ < kMinimumDelay)
    {
        throw std::invalid_argument("DattorroDelay: delay must be at least kMinimumDelay samples");
    }

    if (options.delay_config.lfo_config.has_value())
    {
        SetMod(options.delay_config.lfo_config.value());
    }
}

void DattorroDelay::SetBlend(float blend) noexcept SFFDN_NONBLOCKING
{
    blend_ = blend;
}

float DattorroDelay::GetBlend() const noexcept SFFDN_NONBLOCKING
{
    return blend_;
}

void DattorroDelay::SetFeedforward(float feedforward) noexcept SFFDN_NONBLOCKING
{
    feedforward_ = feedforward;
}

float DattorroDelay::GetFeedforward() const noexcept SFFDN_NONBLOCKING
{
    return feedforward_;
}

void DattorroDelay::SetFeedback(float feedback) noexcept SFFDN_NONBLOCKING
{
    feedback_ = std::clamp(feedback, -kMaxFeedback, kMaxFeedback);
}

float DattorroDelay::GetFeedback() const noexcept SFFDN_NONBLOCKING
{
    return feedback_;
}

void DattorroDelay::SetDelay(float delay)
{
    const float width = lfo_.GetAmplitude();
    if (delay - std::abs(width) < kMinimumDelay)
    {
        throw std::invalid_argument("DattorroDelay::SetDelay: delay minus the modulation width must be at least "
                                    "kMinimumDelay samples");
    }

    const uint32_t required_max_delay = RequiredMaximumDelay(delay, width);
    if (delay_.GetMaximumDelay() < required_max_delay)
    {
        delay_.SetMaximumDelay(required_max_delay);
    }

    base_delay_ = delay;
    feedback_tap_ = static_cast<uint32_t>(std::lround(delay));
    delay_.SetDelay(delay);
}

float DattorroDelay::GetDelay() const
{
    return base_delay_;
}

void DattorroDelay::SetMod(const ModulationOptions& options)
{
    if (base_delay_ - std::abs(options.amplitude) < kMinimumDelay)
    {
        throw std::invalid_argument("DattorroDelay::SetMod: delay minus the modulation width must be at least "
                                    "kMinimumDelay samples");
    }

    const uint32_t required_max_delay = RequiredMaximumDelay(base_delay_, options.amplitude);
    if (delay_.GetMaximumDelay() < required_max_delay)
    {
        delay_.SetMaximumDelay(required_max_delay);
    }

    lfo_.SetFrequency(options.frequency);
    lfo_.SetAmplitude(options.amplitude);
    lfo_.SetPhaseOffset(options.initial_phase);
}

float DattorroDelay::TickInternal(float input, float mod) noexcept SFFDN_NONBLOCKING
{
    delay_.SetDelay(std::max(kMinimumDelay, base_delay_ + mod));

    // Both taps are read before the new sample is written, so that the feedback tap can be used to compute it.
    const float feedback_tap = delay_.TapOut(feedback_tap_ - 1);
    const float feedforward_tap = delay_.NextOut();

    // The feedback is subtracted, matching the negation drawn on the summing junction in the paper. Keeping the
    // negation here rather than folding it into the gain is what lets the presets use the signs of Table 1 verbatim.
    const float delay_input = input - (feedback_ * feedback_tap);
    delay_.Advance(delay_input);

    return (feedforward_ * feedforward_tap) + (blend_ * delay_input);
}

float DattorroDelay::Tick(float input) noexcept SFFDN_NONBLOCKING
{
    return TickInternal(input, lfo_.Tick());
}

void DattorroDelay::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    auto in_span = input.GetChannelSpan(0);
    auto out_span = output.GetChannelSpan(0);

    constexpr uint32_t kUnrollFactor = 16;
    const uint32_t size = in_span.size();
    const uint32_t unroll_size = size & ~(kUnrollFactor - 1);

    uint32_t sample = 0;
    for (; sample < unroll_size; sample += kUnrollFactor)
    {
        std::array<float, kUnrollFactor> mods{};
        lfo_.Generate(mods);

        auto in_batch = in_span.subspan(sample, kUnrollFactor);
        auto out_batch = out_span.subspan(sample, kUnrollFactor);

        for (auto i = 0u; i < kUnrollFactor; ++i)
        {
            out_batch[i] = TickInternal(in_batch[i], mods[i]);
        }
    }

    for (; sample < size; ++sample)
    {
        out_span[sample] = Tick(in_span[sample]);
    }
}

uint32_t DattorroDelay::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

uint32_t DattorroDelay::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return 1;
}

void DattorroDelay::Clear()
{
    // Restore the nominal delay before clearing, so that the delay line does not carry any modulation state over.
    delay_.SetDelay(base_delay_);
    delay_.Clear();
    lfo_.ResetPhase();
}

std::unique_ptr<AudioProcessor> DattorroDelay::Clone() const
{
    return std::make_unique<DattorroDelay>(*this);
}

DattorroDelayOptions MakeDattorroDelayOptions(DattorroEffectType type, float sample_rate)
{
    struct PresetValues
    {
        float blend;
        float feedforward;
        float feedback;
        float delay_ms;
        float width_ms;
        float rate_hz;
        DelayInterpolationType interp_type;
    };

    // The gains are the values given in Table 1 of the paper. The delay, width and rate are nominal values picked
    // inside the ranges given in the paper.
    //
    // The gains are positive, as printed in the table, because the negation of the feedback happens at the summing
    // junction in Process(). For the white chorus that is what the name refers to: with blend = feedback and
    // feedforward = 1 the transfer function (BL + z^-M) / (1 + BL*z^-M) is exactly allpass, so the effect is heard as
    // pure phase modulation with no coloration. Get that sign wrong and the fixed feedback tap becomes a resonant
    // comb with peaks every sample_rate / delay Hz and about 30 dB of ripple, which rings on any harmonic landing on
    // a peak.
    //
    // The modulated presets use linear interpolation rather than allpass. Both are now click free — allpass
    // interpolation used to leave a step in the output every time the delay crossed a sample boundary, but
    // DelayInterp::SetDelay() re-seeds the allpass state from the new tap and the flanger's peak-to-RMS ratio of the
    // second difference measures 1.9 either way (it was 21.0 before that fix). Linear is kept because it is cheaper
    // on the modulated path and its magnitude droop near Nyquist is inaudible for a chorus or a flanger. That
    // reasoning does not carry over to a delay inside a feedback loop, where the droop compounds on every
    // circulation and pulls the high-frequency T60 down; use allpass there. Echo is not modulated, so it needs no
    // interpolation at all.
    constexpr float kSqrtHalf = 0.7071f;
    PresetValues preset{};
    switch (type)
    {
    case DattorroEffectType::Vibrato:
        preset = {.blend = 0.f,
                  .feedforward = 1.f,
                  .feedback = 0.f,
                  .delay_ms = 3.f,
                  .width_ms = 2.f,
                  .rate_hz = 1.f,
                  .interp_type = DelayInterpolationType::Linear};
        break;
    case DattorroEffectType::Flanger:
        preset = {.blend = kSqrtHalf,
                  .feedforward = kSqrtHalf,
                  .feedback = kSqrtHalf,
                  .delay_ms = 1.2f,
                  .width_ms = 1.f,
                  .rate_hz = 0.5f,
                  .interp_type = DelayInterpolationType::Linear};
        break;
    case DattorroEffectType::WhiteChorus:
        preset = {.blend = kSqrtHalf,
                  .feedforward = 1.f,
                  .feedback = kSqrtHalf,
                  .delay_ms = 10.f,
                  .width_ms = 5.f,
                  .rate_hz = 0.15f,
                  .interp_type = DelayInterpolationType::Linear};
        break;
    case DattorroEffectType::Doubling:
        preset = {.blend = kSqrtHalf,
                  .feedforward = kSqrtHalf,
                  .feedback = 0.f,
                  .delay_ms = 30.f,
                  .width_ms = 10.f,
                  .rate_hz = 0.15f,
                  .interp_type = DelayInterpolationType::Linear};
        break;
    case DattorroEffectType::Echo:
        preset = {.blend = 1.f,
                  .feedforward = 1.f,
                  .feedback = 0.7f,
                  .delay_ms = 100.f,
                  .width_ms = 0.f,
                  .rate_hz = 0.f,
                  .interp_type = DelayInterpolationType::None};
        break;
    default:
        assert(false);
        preset = {.blend = 1.f,
                  .feedforward = 0.f,
                  .feedback = 0.f,
                  .delay_ms = 1.f,
                  .width_ms = 0.f,
                  .rate_hz = 0.f,
                  .interp_type = DelayInterpolationType::None};
        break;
    }

    const float samples_per_ms = sample_rate / 1000.f;
    const float delay = std::max(preset.delay_ms * samples_per_ms, DattorroDelay::kMinimumDelay);
    // Keep the modulated tap above the minimum delay, whatever the sample rate is.
    const float width = std::clamp(preset.width_ms * samples_per_ms, 0.f, delay - DattorroDelay::kMinimumDelay);

    DattorroDelayOptions options;
    options.blend = preset.blend;
    options.feedforward = preset.feedforward;
    options.feedback = preset.feedback;
    options.delay_config.delay = delay;
    options.delay_config.max_delay = RequiredMaximumDelay(delay, width);
    options.delay_config.interp_type = preset.interp_type;

    if (width > 0.f && preset.rate_hz > 0.f)
    {
        options.delay_config.lfo_config =
            ModulationOptions{.frequency = preset.rate_hz / sample_rate, .amplitude = width, .initial_phase = 0.f};
    }
    else
    {
        options.delay_config.lfo_config = std::nullopt;
    }

    return options;
}

std::unique_ptr<FilterBank> MakeMultichannelDattorroDelay(const MultichannelDattorroDelayOptions& options)
{
    auto bank = std::make_unique<FilterBank>();
    for (const auto& delay_config : options.delays)
    {
        bank->AddFilter(std::make_unique<DattorroDelay>(delay_config));
    }
    return bank;
}

MultichannelDattorroDelayOptions MakeMultichannelDattorroDelayOptions(DattorroEffectType type, float sample_rate,
                                                                     uint32_t channel_count)
{
    MultichannelDattorroDelayOptions options;
    if (channel_count == 0)
    {
        return options;
    }

    const DattorroDelayOptions base = MakeDattorroDelayOptions(type, sample_rate);

    options.delays.reserve(channel_count);
    for (auto channel = 0u; channel < channel_count; ++channel)
    {
        DattorroDelayOptions channel_options = base;

        // Spread the channels symmetrically around the nominal preset, so that the average across the bank stays on
        // the values of the paper. A single channel gets a spread of exactly 0 and therefore reproduces
        // MakeDattorroDelayOptions() field for field.
        const float spread =
            (channel_count > 1) ? ((static_cast<float>(channel) / static_cast<float>(channel_count - 1)) - 0.5f) * 2.f
                                : 0.f;
        const float scale = 1.f + (kChannelSpread * spread);

        const float delay = std::max(base.delay_config.delay * scale, sfFDN::DattorroDelay::kMinimumDelay);
        channel_options.delay_config.delay = delay;

        if (base.delay_config.lfo_config.has_value())
        {
            const auto& base_lfo = base.delay_config.lfo_config.value();

            // Staggering the initial phase is the primary decorrelator: the channels never reach the extremes of
            // their modulation at the same time.
            const float phase = static_cast<float>(channel) / static_cast<float>(channel_count);
            const float width = std::clamp(base_lfo.amplitude * scale, 0.f, delay - sfFDN::DattorroDelay::kMinimumDelay);

            channel_options.delay_config.lfo_config = ModulationOptions{
                .frequency = base_lfo.frequency * scale, .amplitude = width, .initial_phase = phase};

            // Allpass interpolation rather than the linear interpolation of the single-channel presets: a
            // multichannel bank is meant for the feedback loop, where the magnitude droop of linear interpolation
            // compounds on every circulation.
            channel_options.delay_config.interp_type = DelayInterpolationType::Allpass;
            channel_options.delay_config.max_delay = RequiredMaximumDelay(delay, width);
        }
        else
        {
            channel_options.delay_config.max_delay = RequiredMaximumDelay(delay, 0.f);
        }

        options.delays.push_back(channel_options);
    }

    return options;
}

} // namespace sfFDN
