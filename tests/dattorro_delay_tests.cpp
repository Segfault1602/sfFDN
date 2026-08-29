#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "rng.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr float kSqrtHalf = 0.7071f;

std::vector<float> ImpulseResponse(sfFDN::DattorroDelay& delay, uint32_t size)
{
    std::vector<float> response(size, 0.f);
    response[0] = delay.Tick(1.f);
    for (auto i = 1u; i < size; ++i)
    {
        response[i] = delay.Tick(0.f);
    }
    return response;
}
} // namespace

TEST_CASE("DattorroDelay feedforward only")
{
    constexpr uint32_t kDelay = 8;
    constexpr float kBlend = 0.25f;
    constexpr float kFeedforward = 0.5f;

    sfFDN::DattorroDelay delay(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = static_cast<float>(kDelay),
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::None,
                         .lfo_config = std::nullopt},
        .blend = kBlend,
        .feedforward = kFeedforward,
        .feedback = 0.f,
    });

    const auto response = ImpulseResponse(delay, 32);

    // With no feedback, the impulse response is BL * delta[n] + FF * delta[n - M].
    for (auto n = 0u; n < response.size(); ++n)
    {
        float expected = 0.f;
        if (n == 0)
        {
            expected = kBlend;
        }
        else if (n == kDelay)
        {
            expected = kFeedforward;
        }

        REQUIRE_THAT(response[n], Catch::Matchers::WithinAbs(expected, 1e-6f));
    }
}

TEST_CASE("DattorroDelay feedback echo")
{
    constexpr uint32_t kDelay = 5;
    constexpr float kFeedback = 0.5f;

    sfFDN::DattorroDelay delay(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = static_cast<float>(kDelay),
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::None,
                         .lfo_config = std::nullopt},
        .blend = 1.f,
        .feedforward = 1.f,
        .feedback = kFeedback,
    });

    const auto response = ImpulseResponse(delay, 32);

    // w[n] = delta[n] - FB * w[n - M] gives an impulse train that decays geometrically and alternates in polarity,
    // because the feedback is subtracted at the summing junction. With BL = FF = 1 the output holds two copies of it,
    // offset by M.
    for (auto n = 0u; n < response.size(); ++n)
    {
        float expected = 0.f;
        if (n % kDelay == 0)
        {
            const auto k = n / kDelay;
            expected = std::pow(-kFeedback, static_cast<float>(k));
            if (k > 0)
            {
                expected += std::pow(-kFeedback, static_cast<float>(k - 1));
            }
        }

        REQUIRE_THAT(response[n], Catch::Matchers::WithinAbs(expected, 1e-5f));
    }
}

TEST_CASE("DattorroDelay feedback tap is not modulated")
{
    constexpr uint32_t kDelay = 32;
    constexpr float kFeedback = 0.5f;

    // A wide and fast modulation. Only the feedforward tap should follow it; the recirculation must stay locked to the
    // nominal delay.
    sfFDN::DattorroDelay delay(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = static_cast<float>(kDelay),
                         .max_delay = 128,
                         .interp_type = sfFDN::DelayInterpolationType::Allpass,
                         .lfo_config =
                             sfFDN::ModulationOptions{.frequency = 0.01f, .amplitude = 16.f, .initial_phase = 0.f}},
        .blend = 1.f,
        .feedforward = 0.f,
        .feedback = kFeedback,
    });

    const auto response = ImpulseResponse(delay, 4 * kDelay);

    // With FF = 0 the output is just w[n], which must be an unmodulated, undistorted impulse train.
    for (auto n = 0u; n < response.size(); ++n)
    {
        const float expected = (n % kDelay == 0) ? std::pow(-kFeedback, static_cast<float>(n / kDelay)) : 0.f;
        REQUIRE_THAT(response[n], Catch::Matchers::WithinAbs(expected, 1e-6f));
    }
}

TEST_CASE("DattorroDelay matches SchroederAllpass")
{
    constexpr uint32_t kDelay = 11;
    constexpr float kGain = 0.6f;
    constexpr uint32_t kSampleCount = 128;

    // The Schroeder allpass recursion is w[n] = x[n] + g*w[n-M], and the feedback is subtracted here, so the
    // equivalent settings are FB = -g, FF = 1 and BL = -g, without modulation.
    sfFDN::DattorroDelay delay(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = static_cast<float>(kDelay),
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::None,
                         .lfo_config = std::nullopt},
        .blend = -kGain,
        .feedforward = 1.f,
        .feedback = -kGain,
    });

    sfFDN::SchroederAllpass allpass(kDelay, kGain);

    sfFDN::RNG rng;
    for (auto i = 0u; i < kSampleCount; ++i)
    {
        const float input = rng();
        REQUIRE_THAT(delay.Tick(input), Catch::Matchers::WithinAbs(allpass.Tick(input), 1e-5f));
    }
}

TEST_CASE("DattorroDelay block processing matches Tick")
{
    constexpr uint32_t kBlockSize = 40;
    const sfFDN::DattorroDelayOptions options{
        .delay_config = {.delay = 24.f,
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::Allpass,
                         .lfo_config =
                             sfFDN::ModulationOptions{.frequency = 0.003f, .amplitude = 4.f, .initial_phase = 0.125f}},
        .blend = kSqrtHalf,
        .feedforward = 1.f,
        .feedback = kSqrtHalf,
    };

    std::array<float, kBlockSize> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i) / static_cast<float>(input.size());
    }

    sfFDN::DattorroDelay tick_delay(options);
    std::array<float, kBlockSize> expected{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        expected[i] = tick_delay.Tick(input[i]);
    }

    sfFDN::DattorroDelay block_delay(options);
    std::array<float, kBlockSize> output{};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    block_delay.Process(input_buffer, output_buffer);

    for (const auto [actual, expected_sample] : std::views::zip(output, expected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected_sample, 1e-6f));
    }
}

TEST_CASE("DattorroDelay Clone and Clear")
{
    const sfFDN::DattorroDelayOptions options{
        .delay_config = {.delay = 17.f,
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::Allpass,
                         .lfo_config =
                             sfFDN::ModulationOptions{.frequency = 0.005f, .amplitude = 3.f, .initial_phase = 0.f}},
        .blend = kSqrtHalf,
        .feedforward = 1.f,
        .feedback = kSqrtHalf,
    };

    sfFDN::DattorroDelay delay(options);
    auto clone = delay.Clone();
    REQUIRE(clone != nullptr);
    REQUIRE(clone->InputChannelCount() == 1);
    REQUIRE(clone->OutputChannelCount() == 1);

    sfFDN::RNG rng;
    for (auto i = 0u; i < 64u; ++i)
    {
        const float input = rng();
        std::array<float, 1> clone_input = {input};
        std::array<float, 1> clone_output = {0.f};
        sfFDN::AudioBuffer clone_input_buffer(clone_input);
        sfFDN::AudioBuffer clone_output_buffer(clone_output);
        clone->Process(clone_input_buffer, clone_output_buffer);

        REQUIRE_THAT(clone_output[0], Catch::Matchers::WithinAbs(delay.Tick(input), 1e-6f));
    }

    // Clear() resets the state but keeps the knobs.
    delay.Clear();
    REQUIRE_THAT(delay.GetBlend(), Catch::Matchers::WithinAbs(options.blend, 1e-6f));
    REQUIRE_THAT(delay.GetFeedforward(), Catch::Matchers::WithinAbs(options.feedforward, 1e-6f));
    REQUIRE_THAT(delay.GetFeedback(), Catch::Matchers::WithinAbs(options.feedback, 1e-6f));
    REQUIRE_THAT(delay.GetDelay(), Catch::Matchers::WithinAbs(options.delay_config.delay, 1e-6f));

    sfFDN::DattorroDelay fresh(options);
    for (auto i = 0u; i < 32u; ++i)
    {
        const float input = rng();
        REQUIRE_THAT(delay.Tick(input), Catch::Matchers::WithinAbs(fresh.Tick(input), 1e-6f));
    }
}

TEST_CASE("DattorroDelay does not allocate")
{
    constexpr uint32_t kBlockSize = 64;
    sfFDN::DattorroDelay delay(sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, 48000.f));

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    sfFDN::RNG rng;
    for (float& sample : input)
    {
        sample = rng();
    }

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    // Warm up any lazy state before counting.
    delay.Process(input_buffer, output_buffer);

    {
        const sfFDNTest::ScopedAllocationCounter counter;
        delay.Process(input_buffer, output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

TEST_CASE("DattorroDelay parameter validation")
{
    sfFDN::DattorroDelayOptions options{
        .delay_config = {.delay = 4.f,
                         .max_delay = 64,
                         .interp_type = sfFDN::DelayInterpolationType::Allpass,
                         .lfo_config = std::nullopt},
        .blend = 0.f,
        .feedforward = 1.f,
        .feedback = 0.f,
    };

    // A modulation wider than the nominal delay would push the read pointer past the write pointer.
    options.delay_config.lfo_config =
        sfFDN::ModulationOptions{.frequency = 0.001f, .amplitude = 8.f, .initial_phase = 0.f};
    REQUIRE_THROWS_AS(sfFDN::DattorroDelay(options), std::invalid_argument);

    options.delay_config.delay = 1.f;
    options.delay_config.lfo_config = std::nullopt;
    REQUIRE_THROWS_AS(sfFDN::DattorroDelay(options), std::invalid_argument);

    options.delay_config.delay = 32.f;
    sfFDN::DattorroDelay delay(options);
    REQUIRE_THROWS_AS(
        delay.SetMod(sfFDN::ModulationOptions{.frequency = 0.001f, .amplitude = 31.f, .initial_phase = 0.f}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(delay.SetDelay(1.f), std::invalid_argument);

    // The feedback gain is clamped to keep the loop stable.
    delay.SetFeedback(2.f);
    REQUIRE(delay.GetFeedback() < 1.f);
    delay.SetFeedback(-2.f);
    REQUIRE(delay.GetFeedback() > -1.f);
}

TEST_CASE("DattorroDelay presets")
{
    constexpr float kSampleRate = 48000.f;
    constexpr std::array<sfFDN::DattorroEffectType, 5> kTypes = {
        sfFDN::DattorroEffectType::Vibrato, sfFDN::DattorroEffectType::Flanger, sfFDN::DattorroEffectType::WhiteChorus,
        sfFDN::DattorroEffectType::Doubling, sfFDN::DattorroEffectType::Echo};

    for (const auto type : kTypes)
    {
        const auto options = sfFDN::MakeDattorroDelayOptions(type, kSampleRate);

        const float width =
            options.delay_config.lfo_config.has_value() ? options.delay_config.lfo_config->amplitude : 0.f;
        REQUIRE(options.delay_config.delay - width >= sfFDN::DattorroDelay::kMinimumDelay);
        REQUIRE(options.delay_config.max_delay > options.delay_config.delay + width);
        REQUIRE(std::abs(options.feedback) < 1.f);

        // Every preset must be constructible and produce a finite output.
        sfFDN::DattorroDelay delay(options);
        const auto response = ImpulseResponse(delay, 256);
        for (const float sample : response)
        {
            REQUIRE(std::isfinite(sample));
        }
    }

    // Table 1 of the paper.
    const auto vibrato = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Vibrato, kSampleRate);
    REQUIRE_THAT(vibrato.blend, Catch::Matchers::WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(vibrato.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(vibrato.feedback, Catch::Matchers::WithinAbs(0.f, 1e-6f));

    const auto flanger = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Flanger, kSampleRate);
    REQUIRE_THAT(flanger.blend, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));
    REQUIRE_THAT(flanger.feedforward, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));
    REQUIRE_THAT(flanger.feedback, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));

    const auto chorus = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate);
    REQUIRE_THAT(chorus.blend, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));
    REQUIRE_THAT(chorus.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(chorus.feedback, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));

    const auto doubling = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Doubling, kSampleRate);
    REQUIRE_THAT(doubling.blend, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));
    REQUIRE_THAT(doubling.feedforward, Catch::Matchers::WithinAbs(kSqrtHalf, 1e-4f));
    REQUIRE_THAT(doubling.feedback, Catch::Matchers::WithinAbs(0.f, 1e-6f));

    const auto echo = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Echo, kSampleRate);
    REQUIRE_THAT(echo.blend, Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(echo.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE(echo.feedback > 0.f);
    REQUIRE(!echo.delay_config.lfo_config.has_value());
}

namespace
{
constexpr uint32_t kSampleRate = 48000;

struct PresetInfo
{
    sfFDN::DattorroEffectType type;
    const char* name;
};

constexpr std::array<PresetInfo, 5> kAllPresets = {{
    {sfFDN::DattorroEffectType::Vibrato, "vibrato"},
    {sfFDN::DattorroEffectType::Flanger, "flanger"},
    {sfFDN::DattorroEffectType::WhiteChorus, "white_chorus"},
    {sfFDN::DattorroEffectType::Doubling, "doubling"},
    {sfFDN::DattorroEffectType::Echo, "echo"},
}};

/// @brief Runs a buffer through a processor one block at a time, the way a host would.
std::vector<float> ProcessInBlocks(sfFDN::AudioProcessor& processor, std::span<const float> input, uint32_t block_size)
{
    std::vector<float> output(input.size(), 0.f);
    std::vector<float> block_in(block_size, 0.f);

    for (uint32_t start = 0; start < input.size(); start += block_size)
    {
        const uint32_t count = std::min<uint32_t>(block_size, input.size() - start);

        std::copy_n(input.begin() + start, count, block_in.begin());
        auto in_span = std::span(block_in).first(count);
        auto out_span = std::span(output).subspan(start, count);

        sfFDN::AudioBuffer in_buffer(in_span);
        sfFDN::AudioBuffer out_buffer(out_span);
        processor.Process(in_buffer, out_buffer);
    }

    return output;
}

/// @brief Builds a signal that makes each effect audible: a harmonically rich tone (so the pitch modulation of
/// vibrato and the detuning of chorus can be heard), then white noise (which makes the moving comb notches of the
/// flanger obvious), then an impulse followed by silence (which exposes the echo repeats, and makes any processing
/// click stand out against a silent background).
std::vector<float> MakeAuditionSignal()
{
    constexpr uint32_t kToneSamples = kSampleRate * 3 / 2;
    constexpr uint32_t kNoiseSamples = kSampleRate * 3 / 2;
    constexpr uint32_t kTailSamples = kSampleRate;
    constexpr float kLevel = 0.25f; // Leaves headroom for the feedback gain of the presets.

    std::vector<float> signal(kToneSamples + kNoiseSamples + kTailSamples, 0.f);

    // Band-limited sawtooth at 220 Hz. Summing harmonics explicitly keeps the source free of aliasing, so anything
    // harsh in the output comes from the effect and not from the test signal.
    constexpr float kToneFreq = 220.f;
    const auto kHarmonicCount = static_cast<uint32_t>((kSampleRate / 2.f) / kToneFreq) - 1;
    for (uint32_t harmonic = 1; harmonic <= kHarmonicCount; ++harmonic)
    {
        const float amplitude = 1.f / static_cast<float>(harmonic);
        const float omega = 2.f * std::numbers::pi_v<float> * kToneFreq * static_cast<float>(harmonic) / kSampleRate;
        for (uint32_t i = 0; i < kToneSamples; ++i)
        {
            signal[i] += amplitude * std::sin(omega * static_cast<float>(i));
        }
    }

    const float tone_peak = std::ranges::max(
        std::views::transform(std::span(signal).first(kToneSamples), [](float sample) { return std::abs(sample); }));
    for (uint32_t i = 0; i < kToneSamples; ++i)
    {
        signal[i] *= kLevel / tone_peak;
    }

    sfFDN::RNG rng;
    for (uint32_t i = 0; i < kNoiseSamples; ++i)
    {
        signal[kToneSamples + i] = kLevel * rng();
    }

    signal[kToneSamples + kNoiseSamples] = kLevel * 4.f;

    // Short raised-cosine fades across the two segment boundaries, so the seams of the *input* are smooth. Without
    // them the audition file would contain steps that are trivially mistaken for effect artifacts.
    constexpr uint32_t kFadeSamples = 256;
    for (uint32_t i = 0; i < kFadeSamples; ++i)
    {
        const float ramp = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * static_cast<float>(i) / kFadeSamples));
        signal[i] *= ramp;
        signal[kToneSamples - 1 - i] *= ramp;
        signal[kToneSamples + i] *= ramp;
        signal[kToneSamples + kNoiseSamples - 1 - i] *= ramp;
    }

    return signal;
}

/// @brief Peak-to-RMS ratio of the second difference `y[n-1] - 2*y[n] + y[n+1]` of a signal.
///
/// The second difference is the usual click-detection kernel: it suppresses low frequencies and amplifies abrupt
/// ones. For a sine of amplitude A and normalized frequency w it stays around A*w^2, which is tiny, while a
/// single-sample step of size g produces a peak of 2*g. That gap is what makes a quiet click measurable; the first
/// difference is far blunter, because at audio frequencies the natural slew of the signal is comparable to the click
/// itself.
///
/// A smooth, band-limited signal therefore scores close to sqrt(2), and a discontinuity scores far higher.
float SecondDifferenceCrestFactor(std::span<const float> signal)
{
    double sum_squares = 0.0;
    float max_step = 0.f;
    for (uint32_t i = 1; i + 1 < signal.size(); ++i)
    {
        const float step = std::abs(signal[i - 1] - (2.f * signal[i]) + signal[i + 1]);
        sum_squares += static_cast<double>(step) * step;
        max_step = std::max(max_step, step);
    }

    const auto rms_step = static_cast<float>(std::sqrt(sum_squares / (signal.size() - 2)));
    return (rms_step > 0.f) ? (max_step / rms_step) : 0.f;
}
} // namespace

TEST_CASE("DattorroDelayPresetAudio")
{
    // Deliberately not a multiple of the 16-sample unroll factor inside Process(), so that both the unrolled body and
    // the scalar remainder run every block and any seam between them would show up in the audio.
    constexpr uint32_t kBlockSize = 100;

    const std::vector<float> input = MakeAuditionSignal();
    WriteWavFile("dattorro_input.wav", input);

    for (const auto& [type, name] : kAllPresets)
    {
        sfFDN::DattorroDelay delay(sfFDN::MakeDattorroDelayOptions(type, kSampleRate));
        std::vector<float> output = ProcessInBlocks(delay, input, kBlockSize);

        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));

        // Normalize so every preset is comfortable to listen to at the same volume. The discontinuity test below
        // works on unnormalized output, so this only affects what gets written to disk.
        const float peak =
            std::ranges::max(std::views::transform(output, [](float sample) { return std::abs(sample); }));
        if (peak > 0.f)
        {
            const float gain = 0.9f / peak;
            for (float& sample : output)
            {
                sample *= gain;
            }
        }

        WriteWavFile("dattorro_" + std::string(name) + ".wav", output);
    }
}

TEST_CASE("DattorroDelayPresetContinuity")
{
    // A pure sine is smooth and band-limited, so every output of these presets is a sum of scaled, delayed and
    // interpolated sines and must be smooth too. That makes a click easy to spot: it is an isolated large jump in a
    // sequence of small ones, i.e. an outlier in the first difference.
    constexpr uint32_t kBlockSize = 100;
    constexpr uint32_t kDurationSamples = kSampleRate * 4;
    constexpr float kToneFreq = 440.f;
    constexpr float kLevel = 0.25f;
    constexpr float kMaxCrestFactor = 10.f;

    std::vector<float> input(kDurationSamples, 0.f);
    const float omega = 2.f * std::numbers::pi_v<float> * kToneFreq / kSampleRate;
    for (uint32_t i = 0; i < kDurationSamples; ++i)
    {
        input[i] = kLevel * std::sin(omega * static_cast<float>(i));
    }

    // Fade the tone in. A delay line starts full of zeros, so without a fade the moment the delayed copy first
    // arrives is a genuine step - inherent to any delay, not a defect - and it would dominate the measurement.
    constexpr uint32_t kFadeSamples = 2048;
    for (uint32_t i = 0; i < kFadeSamples; ++i)
    {
        input[i] *= 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * static_cast<float>(i) / kFadeSamples));
    }

    for (const auto& [type, name] : kAllPresets)
    {
        INFO("preset: " << name);

        sfFDN::DattorroDelay delay(sfFDN::MakeDattorroDelayOptions(type, kSampleRate));
        std::vector<float> output = ProcessInBlocks(delay, input, kBlockSize);

        const float crest_factor = SecondDifferenceCrestFactor(output);
        UNSCOPED_INFO("crest factor " << crest_factor);

        // A smooth sinusoidal signal has a first-difference crest factor of sqrt(2); a few superimposed delayed
        // copies with slightly different instantaneous frequencies push it to about 3. A click would put it in the
        // hundreds. The bound is loose on purpose: it should only ever fire for a real discontinuity.
        REQUIRE(crest_factor < kMaxCrestFactor);

        // Prove the measure above can actually fail. Injecting a single-sample step of 2% of the peak level - far
        // quieter than any glitch worth hearing - must push the same signal over the threshold.
        const float peak =
            std::ranges::max(std::views::transform(output, [](float sample) { return std::abs(sample); }));
        output[output.size() / 2] += 0.02f * peak;
        REQUIRE(SecondDifferenceCrestFactor(output) > kMaxCrestFactor);
    }
}

TEST_CASE("DattorroDelayChorusSine")
{
    // The white chorus LFO runs at 0.15 Hz, so one modulation cycle lasts about 6.7 s. The file is long enough to
    // hear two full cycles sweep past.
    constexpr uint32_t kBlockSize = 100;
    constexpr uint32_t kDurationSamples = kSampleRate * 14;
    constexpr float kToneFreq = 440.f;
    constexpr float kLevel = 0.25f;
    constexpr uint32_t kFadeSamples = 4096;

    std::vector<float> input(kDurationSamples, 0.f);
    const float omega = 2.f * std::numbers::pi_v<float> * kToneFreq / kSampleRate;
    for (uint32_t i = 0; i < kDurationSamples; ++i)
    {
        input[i] = kLevel * std::sin(omega * static_cast<float>(i));
    }
    for (uint32_t i = 0; i < kFadeSamples; ++i)
    {
        const float ramp = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * static_cast<float>(i) / kFadeSamples));
        input[i] *= ramp;
        input[kDurationSamples - 1 - i] *= ramp;
    }

    sfFDN::DattorroDelay chorus(sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate));
    const std::vector<float> output = ProcessInBlocks(chorus, input, kBlockSize);

    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));

    WriteWavFile("dattorro_chorus_sine_dry.wav", input);
    WriteWavFile("dattorro_chorus_sine.wav", output);

    // A chorus on a steady tone is heard as a swirl. The delayed voice is Doppler-shifted by the moving tap (+/-0.47%
    // here, about 8 cents) and beats against the dry path, so the envelope has to actually move. Note the beating is
    // much faster than the 0.15 Hz LFO: as the tap sweeps between 5 and 15 ms the comb notches cross the 440 Hz tone
    // several times per LFO cycle, which measures out at roughly 1.8 Hz. Comparing the loudest and quietest
    // half-second, past the fade-in and stopping before the fade-out, proves the effect is doing something audible
    // rather than passing the tone through. Half-second windows partly average the beating out, so the bound is
    // deliberately well below the ~15 dB swing the instantaneous envelope actually shows.
    constexpr uint32_t kWindow = kSampleRate / 2;
    float min_rms = std::numeric_limits<float>::max();
    float max_rms = 0.f;
    for (uint32_t start = kSampleRate; start + kWindow < kDurationSamples - kSampleRate; start += kWindow)
    {
        double sum_squares = 0.0;
        for (uint32_t i = start; i < start + kWindow; ++i)
        {
            sum_squares += static_cast<double>(output[i]) * output[i];
        }
        const auto rms = static_cast<float>(std::sqrt(sum_squares / kWindow));
        min_rms = std::min(min_rms, rms);
        max_rms = std::max(max_rms, rms);
    }

    UNSCOPED_INFO("envelope swing " << (20.f * std::log10(max_rms / min_rms)) << " dB");
    REQUIRE(max_rms > min_rms * 1.2f);

    // And it must stay smooth while doing it.
    REQUIRE(SecondDifferenceCrestFactor(output) < 10.f);
}

TEST_CASE("DattorroDelayWhiteChorusIsWhite")
{
    // The "white" in white chorus means a flat magnitude response: with blend = feedback and feedforward = 1 the
    // transfer function (BL + z^-M) / (1 + BL*z^-M) is allpass, so the effect is pure phase modulation and adds no
    // coloration. Get the sign of the feedback wrong and the fixed feedback tap becomes a resonant comb with peaks
    // every sample_rate / delay Hz, which rings audibly on any harmonic sitting on a peak.
    //
    // The check is on the static response, with the LFO switched off, because it is the *fixed* feedback tap that
    // sets the magnitude response; the modulation only moves the phase around.
    auto options = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate);

    REQUIRE_THAT(options.blend, Catch::Matchers::WithinAbs(options.feedback, 1e-6f));
    REQUIRE_THAT(options.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-6f));

    options.delay_config.lfo_config = std::nullopt;
    sfFDN::DattorroDelay chorus(options);

    // Measure the magnitude response from the impulse response. The delay is 480 samples and the feedback decays by
    // 0.7071 per pass, so a few thousand samples captures it to well below the tolerance used here.
    constexpr uint32_t kFftSize = 1 << 15;
    std::vector<float> impulse(kFftSize, 0.f);
    impulse[0] = chorus.Tick(1.f);
    for (uint32_t i = 1; i < kFftSize; ++i)
    {
        impulse[i] = chorus.Tick(0.f);
    }

    // Naive DFT at a spread of frequencies, including the comb peak positions that a wrong sign would light up.
    const float bin_spacing = kSampleRate / static_cast<float>(options.delay_config.delay);
    float min_gain = std::numeric_limits<float>::max();
    float max_gain = 0.f;
    for (uint32_t step = 1; step <= 60; ++step)
    {
        // Half-integer multiples land on the troughs, integer multiples on the peaks.
        const float freq = 0.5f * static_cast<float>(step) * bin_spacing;
        if (freq >= kSampleRate / 2.f)
        {
            break;
        }

        const float omega = 2.f * std::numbers::pi_v<float> * freq / kSampleRate;
        double real = 0.0;
        double imag = 0.0;
        for (uint32_t i = 0; i < kFftSize; ++i)
        {
            real += impulse[i] * std::cos(omega * static_cast<float>(i));
            imag -= impulse[i] * std::sin(omega * static_cast<float>(i));
        }

        const auto gain = static_cast<float>(std::sqrt((real * real) + (imag * imag)));
        min_gain = std::min(min_gain, gain);
        max_gain = std::max(max_gain, gain);
    }

    const float ripple_db = 20.f * std::log10(max_gain / min_gain);
    UNSCOPED_INFO("ripple " << ripple_db << " dB (min " << min_gain << ", max " << max_gain << ")");

    // A true allpass is flat to within the truncation error of the impulse response. A sign error gives ~30 dB.
    REQUIRE(ripple_db < 1.f);
}

TEST_CASE("MultichannelDattorroDelay channel count")
{
    sfFDN::MultichannelDattorroDelayOptions options;
    REQUIRE(sfFDN::MakeMultichannelDattorroDelay(options)->InputChannelCount() == 0);

    constexpr uint32_t kChannelCount = 6;
    options = sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, 48000.f,
                                                          kChannelCount);
    REQUIRE(options.delays.size() == kChannelCount);

    auto bank = sfFDN::MakeMultichannelDattorroDelay(options);
    REQUIRE(bank->InputChannelCount() == kChannelCount);
    REQUIRE(bank->OutputChannelCount() == kChannelCount);
}

TEST_CASE("MultichannelDattorroDelay per-channel independence")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 64;
    constexpr uint32_t kBlockCount = 8;

    const auto options =
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::Flanger, 48000.f, kChannelCount);

    auto bank = sfFDN::MakeMultichannelDattorroDelay(options);

    // One standalone processor per channel, built from the same config, to compare against.
    std::vector<sfFDN::DattorroDelay> references;
    references.reserve(kChannelCount);
    for (const auto& channel_options : options.delays)
    {
        references.emplace_back(channel_options);
    }

    sfFDN::RNG rng;
    std::vector<float> input(static_cast<size_t>(kBlockSize) * kChannelCount, 0.f);
    std::vector<float> output(input.size(), 0.f);

    for (auto block = 0u; block < kBlockCount; ++block)
    {
        for (float& sample : input)
        {
            sample = rng();
        }

        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);
        bank->Process(input_buffer, output_buffer);

        for (auto channel = 0u; channel < kChannelCount; ++channel)
        {
            const auto in_span = input_buffer.GetChannelSpan(channel);
            const auto out_span = output_buffer.GetChannelSpan(channel);
            for (auto sample = 0u; sample < kBlockSize; ++sample)
            {
                const float expected = references[channel].Tick(in_span[sample]);
                REQUIRE_THAT(out_span[sample], Catch::Matchers::WithinAbs(expected, 1e-6f));
            }
        }
    }
}

TEST_CASE("MultichannelDattorroDelayOptions decorrelation")
{
    constexpr uint32_t kChannelCount = 8;
    constexpr float kSampleRate = 48000.f;

    const auto base = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate);
    const auto options =
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate,
                                                    kChannelCount);
    REQUIRE(options.delays.size() == kChannelCount);

    std::vector<float> phases;
    for (const auto& channel_options : options.delays)
    {
        // The gains are the values of the paper and are shared by every channel.
        REQUIRE_THAT(channel_options.blend, Catch::Matchers::WithinAbs(base.blend, 1e-6f));
        REQUIRE_THAT(channel_options.feedforward, Catch::Matchers::WithinAbs(base.feedforward, 1e-6f));
        REQUIRE_THAT(channel_options.feedback, Catch::Matchers::WithinAbs(base.feedback, 1e-6f));

        // Modulated channels use allpass interpolation, so that the magnitude response stays flat inside a feedback
        // loop.
        REQUIRE(channel_options.delay_config.interp_type == sfFDN::DelayInterpolationType::Allpass);

        REQUIRE(channel_options.delay_config.lfo_config.has_value());
        const auto& lfo = channel_options.delay_config.lfo_config.value();
        REQUIRE(lfo.initial_phase >= 0.f);
        REQUIRE(lfo.initial_phase < 1.f);
        REQUIRE(lfo.frequency > 0.f);

        // The modulated tap must stay above the minimum delay, and the buffer must be able to hold it.
        REQUIRE(channel_options.delay_config.delay - lfo.amplitude >= sfFDN::DattorroDelay::kMinimumDelay);
        REQUIRE(static_cast<float>(channel_options.delay_config.max_delay) >=
                channel_options.delay_config.delay + lfo.amplitude);

        phases.push_back(lfo.initial_phase);
    }

    // Every channel starts its modulation at a different point in the cycle.
    std::ranges::sort(phases);
    REQUIRE(std::ranges::adjacent_find(phases) == phases.end());

    // The nominal delay and the LFO rate are spread across the bank rather than shared.
    REQUIRE(options.delays.front().delay_config.delay < options.delays.back().delay_config.delay);
    REQUIRE(options.delays.front().delay_config.lfo_config->frequency <
            options.delays.back().delay_config.lfo_config->frequency);

    // A single channel reproduces the single-channel preset exactly, spread and all.
    const auto mono = sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate,
                                                                  1);
    REQUIRE(mono.delays.size() == 1);
    REQUIRE_THAT(mono.delays[0].delay_config.delay, Catch::Matchers::WithinAbs(base.delay_config.delay, 1e-6f));
    REQUIRE_THAT(mono.delays[0].delay_config.lfo_config->amplitude,
                 Catch::Matchers::WithinAbs(base.delay_config.lfo_config->amplitude, 1e-6f));
    REQUIRE_THAT(mono.delays[0].delay_config.lfo_config->frequency,
                 Catch::Matchers::WithinAbs(base.delay_config.lfo_config->frequency, 1e-9f));
    REQUIRE_THAT(mono.delays[0].delay_config.lfo_config->initial_phase, Catch::Matchers::WithinAbs(0.f, 1e-6f));

    // Echo is not modulated, so it needs no interpolation.
    const auto echo = sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::Echo, kSampleRate, 4);
    for (const auto& channel_options : echo.delays)
    {
        REQUIRE_FALSE(channel_options.delay_config.lfo_config.has_value());
        REQUIRE(channel_options.delay_config.interp_type == sfFDN::DelayInterpolationType::None);
    }
}

TEST_CASE("MultichannelDattorroDelay does not allocate")
{
    constexpr uint32_t kChannelCount = 8;
    constexpr uint32_t kBlockSize = 64;

    auto bank = sfFDN::MakeMultichannelDattorroDelay(
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, 48000.f, kChannelCount));

    sfFDN::RNG rng;
    std::vector<float> input(static_cast<size_t>(kBlockSize) * kChannelCount, 0.f);
    for (float& sample : input)
    {
        sample = rng();
    }
    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    // Warm up any lazy state before counting.
    bank->Process(input_buffer, output_buffer);

    {
        const sfFDNTest::ScopedAllocationCounter counter;
        bank->Process(input_buffer, output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

TEST_CASE("MultichannelDattorroDelay clone")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 32;

    auto bank = sfFDN::MakeMultichannelDattorroDelay(
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::Flanger, 48000.f, kChannelCount));

    sfFDN::RNG rng;
    std::vector<float> input(static_cast<size_t>(kBlockSize) * kChannelCount, 0.f);
    for (float& sample : input)
    {
        sample = rng();
    }
    std::vector<float> original_output(input.size(), 0.f);
    std::vector<float> clone_output(input.size(), 0.f);

    // Run a few blocks first so that the clone has non-trivial delay line and LFO state to carry over.
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, original_output);
        for (auto block = 0u; block < 4u; ++block)
        {
            bank->Process(input_buffer, output_buffer);
        }
    }

    auto clone = bank->Clone();
    REQUIRE(clone->InputChannelCount() == kChannelCount);
    REQUIRE(clone->OutputChannelCount() == kChannelCount);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer original_buffer(kBlockSize, kChannelCount, original_output);
    sfFDN::AudioBuffer clone_buffer(kBlockSize, kChannelCount, clone_output);

    bank->Process(input_buffer, original_buffer);
    clone->Process(input_buffer, clone_buffer);

    for (auto i = 0u; i < original_output.size(); ++i)
    {
        REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(original_output[i], 1e-6f));
    }
}

TEST_CASE("DattorroDelay preset gain in a feedback loop")
{
    constexpr float kSampleRate = 48000.f;

    // Frozen-time magnitude response of the comb, with the feedforward tap displaced by `mod` samples from the fixed
    // feedback tap: H(z) = (BL + FF * z^-(M + mod)) / (1 + FB * z^-M).
    const auto peak_gain = [](const sfFDN::DattorroDelayOptions& options, float mod) {
        constexpr uint32_t kBins = 20000;
        const auto m = static_cast<double>(options.delay_config.delay);
        float peak = 0.f;
        for (auto k = 1u; k < kBins; ++k)
        {
            const double w = std::numbers::pi * k / kBins;
            const std::complex<double> num =
                static_cast<double>(options.blend) +
                (static_cast<double>(options.feedforward) * std::polar(1.0, -w * (m + mod)));
            const std::complex<double> den = 1.0 + (static_cast<double>(options.feedback) * std::polar(1.0, -w * m));
            peak = std::max(peak, static_cast<float>(std::abs(num / den)));
        }
        return peak;
    };

    const auto width = [](const sfFDN::DattorroDelayOptions& options) {
        return options.delay_config.lfo_config.has_value() ? options.delay_config.lfo_config->amplitude : 0.f;
    };

    // Vibrato has no feedback and no blend, so it is a pure modulated delay: unity gain at every frequency and at
    // every point of the modulation cycle. It is the only modulated preset that is safe inside a feedback loop.
    {
        const auto vibrato = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Vibrato, kSampleRate);
        const float w = width(vibrato);
        REQUIRE(w > 0.f);
        for (const float mod : {0.f, 0.5f, -0.5f, w, -w})
        {
            REQUIRE_THAT(peak_gain(vibrato, mod), Catch::Matchers::WithinAbs(1.f, 1e-3f));
        }
    }

    // Doubling has no feedback either, so modulating the tap cannot change its gain. It is a fixed +3 dB.
    {
        const auto doubling = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Doubling, kSampleRate);
        const float w = width(doubling);
        for (const float mod : {0.f, w, -w})
        {
            REQUIRE_THAT(peak_gain(doubling, mod),
                         Catch::Matchers::WithinAbs(doubling.blend + doubling.feedforward, 1e-3f));
        }
    }

    // WhiteChorus is allpass only while the two taps coincide. The pole sits at the fixed feedback tap and the zero
    // moves with the feedforward tap, so the cancellation fails as soon as the tap is modulated at all. Half a sample
    // of displacement is already worth more than 10 dB, and the gain saturates at the (BL + FF) / (1 - FB) bound well
    // before the tap reaches the edge of its excursion. This is why a chorus cannot be dropped into an FDN feedback
    // loop.
    {
        const auto chorus = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate);
        const float bound = (chorus.blend + chorus.feedforward) / (1.f - chorus.feedback);

        REQUIRE_THAT(peak_gain(chorus, 0.f), Catch::Matchers::WithinAbs(1.f, 1e-3f));
        REQUIRE(peak_gain(chorus, 0.5f) > 3.f);
        REQUIRE(peak_gain(chorus, width(chorus)) > 4.f);

        for (const float mod : {0.f, 0.5f, 2.f, width(chorus)})
        {
            REQUIRE(peak_gain(chorus, mod) <= bound + 1e-3f);
        }
    }

    // The Flanger carries the same feedback and is likewise well above unity once modulated.
    {
        const auto flanger = sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::Flanger, kSampleRate);
        const float bound = (flanger.blend + flanger.feedforward) / (1.f - flanger.feedback);
        REQUIRE(peak_gain(flanger, width(flanger)) > 4.f);
        REQUIRE(peak_gain(flanger, width(flanger)) <= bound + 1e-3f);
    }
}
