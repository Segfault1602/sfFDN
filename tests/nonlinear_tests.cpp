#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "dc_blocker.h"
#include "rng.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
constexpr float kSampleRate = 96000.f;
constexpr float kSqrt2 = std::numbers::sqrt2_v<float>;

/** @brief A direct transcription of the reference `cfwr` and `DCBlocker` of the DAFx26 companion repository, kept
 * independent of the library implementation so that it can be used as a reference to compare against.
 *
 * Reference: gdalsanto/shimmer-fdn-reverb, src/nonlinearities.py.
 */
class ReferenceRectifier
{
  public:
    ReferenceRectifier(float alpha, bool antialiasing, bool dc_block, float sample_rate)
        : alpha_(alpha)
        , gain_(std::sqrt(2.f - (2.f * std::abs(alpha - 0.5f))))
        , antialiasing_(antialiasing)
        , dc_block_(dc_block)
        , envelope_coeff_(std::exp(-1.f / (sample_rate * 0.05f)))
        , gain_coeff_(std::exp(-1.f / (sample_rate * 0.02f)))
    {
    }

    float operator()(float x)
    {
        float rectified = std::abs(x);
        if (antialiasing_)
        {
            const float den = x - prev_input_;
            if (std::abs(den) <= sfFDN::ControllableFullWaveRectifier::kAntialiasingEpsilon)
            {
                rectified = std::abs(x + prev_input_) / 2.f;
            }
            else
            {
                rectified = ((0.5f * x * std::abs(x)) - (0.5f * prev_input_ * std::abs(prev_input_))) / den;
            }
        }
        prev_input_ = x;

        float y = gain_ * ((alpha_ * rectified) + ((1.f - alpha_) * x));

        if (dc_block_)
        {
            const float blocked = y - prev_blocker_input_ + (0.995f * prev_blocker_output_);
            prev_blocker_input_ = y;
            prev_blocker_output_ = blocked;

            in_pow_ = (envelope_coeff_ * in_pow_) + ((1.f - envelope_coeff_) * y * y);
            out_pow_ = (envelope_coeff_ * out_pow_) + ((1.f - envelope_coeff_) * blocked * blocked);

            const float target = std::min(std::sqrt((in_pow_ + 1e-12f) / (out_pow_ + 1e-12f)), 4.f);
            makeup_ = (gain_coeff_ * makeup_) + ((1.f - gain_coeff_) * target);

            y = makeup_ * blocked;
        }

        return y;
    }

  private:
    float alpha_;
    float gain_;
    bool antialiasing_;
    bool dc_block_;
    float envelope_coeff_;
    float gain_coeff_;

    float prev_input_{0.f};
    float prev_blocker_input_{0.f};
    float prev_blocker_output_{0.f};
    float in_pow_{1e-12f};
    float out_pow_{1e-12f};
    float makeup_{1.f};
};

/** @brief A direct transcription of the reference `SDFD` of the DAFx26 companion repository. */
class ReferenceSdfd
{
  public:
    explicit ReferenceSdfd(float d)
        : d_(d)
    {
    }

    float operator()(float x)
    {
        const float p = std::max(x, 0.f);
        const float n = std::min(x, 0.f);

        const float s1 = ((1.f - d_) * n1_) + (d_ * n);
        const float y = s1 + (d_ * p2_) + ((1.f - d_) * p1_);

        p2_ = p1_;
        p1_ = p;
        n1_ = n;

        return y;
    }

  private:
    float d_;
    float p1_{0.f};
    float p2_{0.f};
    float n1_{0.f};
};

std::vector<float> MakeNoise(uint32_t count, uint32_t seed = 12345)
{
    sfFDN::RNG rng(seed);
    std::vector<float> noise(count, 0.f);
    for (auto& sample : noise)
    {
        sample = rng();
    }
    return noise;
}

std::vector<float> MakeSine(uint32_t count, float normalized_frequency, float amplitude = 0.5f)
{
    std::vector<float> sine(count, 0.f);
    for (auto i = 0u; i < count; ++i)
    {
        sine[i] = amplitude * std::sin(2.f * std::numbers::pi_v<float> * normalized_frequency * static_cast<float>(i));
    }
    return sine;
}

std::vector<float> ProcessBlock(sfFDN::AudioProcessor& processor, std::span<const float> input)
{
    std::vector<float> input_copy(input.begin(), input.end());
    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(input_copy);
    sfFDN::AudioBuffer output_buffer(output);
    processor.Process(input_buffer, output_buffer);

    return output;
}

float Energy(std::span<const float> signal)
{
    float energy = 0.f;
    for (const float sample : signal)
    {
        energy += sample * sample;
    }
    return energy;
}
} // namespace

// ==================== ControllableFullWaveRectifier ====================

TEST_CASE("ControllableFullWaveRectifier matches the reference implementation")
{
    const auto alpha = GENERATE(0.f, 0.25f, 0.5f, 1.f);
    const auto input = MakeNoise(512);

    for (const bool antialiasing : {false, true})
    {
        for (const bool dc_block : {false, true})
        {
            sfFDN::ControllableFullWaveRectifier rectifier(
                sfFDN::ControllableFullWaveRectifierOptions{.alpha = alpha,
                                                            .antialiasing = antialiasing,
                                                            .dc_block = dc_block,
                                                            .sample_rate = kSampleRate});
            ReferenceRectifier reference(alpha, antialiasing, dc_block, kSampleRate);

            const auto output = ProcessBlock(rectifier, input);
            for (auto i = 0u; i < input.size(); ++i)
            {
                // The tolerance is loose for a float32 comparison because the two implementations are in different
                // translation units and the compiler is free to contract their multiply-adds differently. The dc
                // blocker then amplifies that divergence: its envelope recursions have a pole at exp(-1/(fs*tau)),
                // which is 0.9998 here.
                REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(reference(input[i]), 1e-4f));
            }
        }
    }
}

TEST_CASE("ControllableFullWaveRectifier compensation gain")
{
    // Equation (3) of the paper: the gain is one where the operation is exactly energy preserving, and peaks at
    // sqrt(2) where the rectifier becomes a half-wave rectifier.
    sfFDN::ControllableFullWaveRectifier rectifier;

    rectifier.SetAlpha(0.f);
    REQUIRE_THAT(rectifier.GetCompensationGain(), Catch::Matchers::WithinAbs(1.f, 1e-6f));

    rectifier.SetAlpha(1.f);
    REQUIRE_THAT(rectifier.GetCompensationGain(), Catch::Matchers::WithinAbs(1.f, 1e-6f));

    rectifier.SetAlpha(0.5f);
    REQUIRE_THAT(rectifier.GetCompensationGain(), Catch::Matchers::WithinAbs(kSqrt2, 1e-6f));
}

TEST_CASE("ControllableFullWaveRectifier alpha endpoints")
{
    const auto input = MakeNoise(256);

    // alpha = 0 leaves the signal untouched, whatever the antialiasing does.
    sfFDN::ControllableFullWaveRectifier identity(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 0.f, .antialiasing = true, .dc_block = false, .sample_rate = kSampleRate});
    const auto passed = ProcessBlock(identity, input);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(passed[i], Catch::Matchers::WithinAbs(input[i], 1e-6f));
    }

    // alpha = 1 without antialiasing is exactly the full-wave rectifier.
    sfFDN::ControllableFullWaveRectifier rectifier(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = false, .dc_block = false, .sample_rate = kSampleRate});
    const auto rectified = ProcessBlock(rectifier, input);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(rectified[i], Catch::Matchers::WithinAbs(std::abs(input[i]), 1e-6f));
    }
}

TEST_CASE("ControllableFullWaveRectifier antialiasing lowers the aliased noise floor")
{
    // A rectifier generates every even harmonic of its input, and the ones above Nyquist fold back between the
    // harmonics. Section 3.1 of the paper. Picking a fundamental that is not a submultiple of the sample rate means
    // the folded partials do not land on the harmonics, so the energy sitting off the harmonic grid measures the
    // aliasing directly.
    constexpr uint32_t kSize = 8192;
    constexpr float kFrequency = 1760.f / kSampleRate;
    const auto input = MakeSine(kSize, kFrequency);

    auto residual_energy = [&](bool antialiasing) {
        sfFDN::ControllableFullWaveRectifier rectifier(sfFDN::ControllableFullWaveRectifierOptions{
            .alpha = 1.f, .antialiasing = antialiasing, .dc_block = false, .sample_rate = kSampleRate});
        const auto output = ProcessBlock(rectifier, input);

        // Project out the first sixteen harmonics of the rectified sine and measure what is left over.
        std::vector<float> residual(output.begin(), output.end());
        const float mean = std::reduce(residual.begin(), residual.end(), 0.f) / static_cast<float>(kSize);
        for (float& sample : residual)
        {
            sample -= mean;
        }

        for (auto harmonic = 1u; harmonic <= 16u; ++harmonic)
        {
            const float harmonic_frequency = kFrequency * static_cast<float>(harmonic);
            float real = 0.f;
            float imag = 0.f;
            for (auto i = 0u; i < kSize; ++i)
            {
                const float phase = 2.f * std::numbers::pi_v<float> * harmonic_frequency * static_cast<float>(i);
                real += residual[i] * std::cos(phase);
                imag += residual[i] * std::sin(phase);
            }
            real *= 2.f / static_cast<float>(kSize);
            imag *= 2.f / static_cast<float>(kSize);

            for (auto i = 0u; i < kSize; ++i)
            {
                const float phase = 2.f * std::numbers::pi_v<float> * harmonic_frequency * static_cast<float>(i);
                residual[i] -= (real * std::cos(phase)) + (imag * std::sin(phase));
            }
        }

        return Energy(residual);
    };

    REQUIRE(residual_energy(true) < residual_energy(false));
}

TEST_CASE("ControllableFullWaveRectifier antialiasing is continuous across the fallback")
{
    // A constant input drives the denominator of equation (4) to exactly zero on every sample but the first, so the
    // ill-conditioned branch is taken. It must agree with the rectifier it approximates.
    constexpr float kLevel = 0.37f;
    sfFDN::ControllableFullWaveRectifier rectifier(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = false, .sample_rate = kSampleRate});

    rectifier.Tick(kLevel);
    for (auto i = 0u; i < 8u; ++i)
    {
        REQUIRE_THAT(rectifier.Tick(kLevel), Catch::Matchers::WithinAbs(kLevel, 1e-6f));
    }

    // A ramp that steps just under the threshold also lands on the fallback, and must stay close to |x|. The first
    // sample is skipped: the antiderivative approximation is the *average* of |x| over the step from the previous
    // sample, so a jump from silence to kLevel correctly reports half of it.
    sfFDN::ControllableFullWaveRectifier ramped(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = false, .sample_rate = kSampleRate});
    const float step = sfFDN::ControllableFullWaveRectifier::kAntialiasingEpsilon * 0.5f;
    float value = kLevel;
    ramped.Tick(value);
    for (auto i = 0u; i < 16u; ++i)
    {
        value += step;
        const float output = ramped.Tick(value);
        REQUIRE_THAT(output, Catch::Matchers::WithinAbs(std::abs(value), 1e-4f));
    }
}

TEST_CASE("ControllableFullWaveRectifier energy is nearly constant across alpha")
{
    // Section 4.2 of the paper: the uncompensated power varies quadratically with alpha and dips at alpha = 0.5, and
    // g_cfwr corrects that variation. It is exact at alpha = 0, 0.5 and 1, and slightly under one in between.
    const auto input = MakeSine(4096, 1760.f / kSampleRate);
    const float input_energy = Energy(input);

    auto energy_ratio = [&](float alpha) {
        sfFDN::ControllableFullWaveRectifier rectifier(sfFDN::ControllableFullWaveRectifierOptions{
            .alpha = alpha, .antialiasing = false, .dc_block = false, .sample_rate = kSampleRate});
        return Energy(ProcessBlock(rectifier, input)) / input_energy;
    };

    for (const float alpha : {0.f, 0.5f, 1.f})
    {
        REQUIRE_THAT(energy_ratio(alpha), Catch::Matchers::WithinAbs(1.f, 1e-3f));
    }

    // A quarter of the way in, the analysis predicts (1 + (1 - 2*alpha)^2) / 2 * (2 - 2*|alpha - 0.5|) = 0.9375.
    for (const float alpha : {0.25f, 0.75f})
    {
        REQUIRE_THAT(energy_ratio(alpha), Catch::Matchers::WithinAbs(0.9375f, 5e-3f));
    }
}

TEST_CASE("ControllableFullWaveRectifier dc blocker removes the offset")
{
    const auto input = MakeSine(48000, 1000.f / kSampleRate);

    sfFDN::ControllableFullWaveRectifier blocked(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = true, .sample_rate = kSampleRate});
    sfFDN::ControllableFullWaveRectifier unblocked(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = false, .sample_rate = kSampleRate});

    const auto blocked_output = ProcessBlock(blocked, input);
    const auto unblocked_output = ProcessBlock(unblocked, input);

    auto mean = [](std::span<const float> signal) {
        // Skip the transient of the blocker and of the make-up gain.
        const auto tail = signal.subspan(signal.size() / 2);
        return std::reduce(tail.begin(), tail.end(), 0.f) / static_cast<float>(tail.size());
    };

    // A rectified sine is all positive, so its mean is large. The blocker must bring it close to zero.
    REQUIRE(mean(unblocked_output) > 0.2f);
    REQUIRE(std::abs(mean(blocked_output)) < 1e-2f);
}

TEST_CASE("ControllableFullWaveRectifier block processing matches Tick")
{
    const sfFDN::ControllableFullWaveRectifierOptions options{
        .alpha = 0.75f, .antialiasing = true, .dc_block = true, .sample_rate = kSampleRate};

    sfFDN::ControllableFullWaveRectifier block_processor(options);
    sfFDN::ControllableFullWaveRectifier tick_processor(options);

    const auto input = MakeNoise(256, 999);
    const auto block_output = ProcessBlock(block_processor, input);

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(block_output[i], Catch::Matchers::WithinAbs(tick_processor.Tick(input[i]), 1e-6f));
    }
}

TEST_CASE("ControllableFullWaveRectifier Clone and Clear")
{
    const sfFDN::ControllableFullWaveRectifierOptions options{
        .alpha = 0.4f, .antialiasing = true, .dc_block = true, .sample_rate = kSampleRate};

    sfFDN::ControllableFullWaveRectifier rectifier(options);

    // Warm the state up first, so that the clone has something to carry over.
    const auto warmup = MakeNoise(64, 7);
    for (const float sample : warmup)
    {
        rectifier.Tick(sample);
    }

    auto clone = rectifier.Clone();
    REQUIRE(clone != nullptr);
    REQUIRE(clone->InputChannelCount() == 1);
    REQUIRE(clone->OutputChannelCount() == 1);

    const auto input = MakeNoise(128, 31);
    for (const float sample : input)
    {
        std::array<float, 1> clone_input = {sample};
        std::array<float, 1> clone_output = {0.f};
        sfFDN::AudioBuffer clone_input_buffer(clone_input);
        sfFDN::AudioBuffer clone_output_buffer(clone_output);
        clone->Process(clone_input_buffer, clone_output_buffer);

        REQUIRE_THAT(clone_output[0], Catch::Matchers::WithinAbs(rectifier.Tick(sample), 1e-6f));
    }

    // Clear() resets the state but keeps the configuration.
    rectifier.Clear();
    REQUIRE_THAT(rectifier.GetAlpha(), Catch::Matchers::WithinAbs(options.alpha, 1e-6f));

    sfFDN::ControllableFullWaveRectifier fresh(options);
    for (const float sample : input)
    {
        REQUIRE_THAT(rectifier.Tick(sample), Catch::Matchers::WithinAbs(fresh.Tick(sample), 1e-6f));
    }
}

TEST_CASE("ControllableFullWaveRectifier parameter validation")
{
    REQUIRE_THROWS_AS(sfFDN::ControllableFullWaveRectifier(
                          sfFDN::ControllableFullWaveRectifierOptions{.alpha = -0.1f, .sample_rate = kSampleRate}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::ControllableFullWaveRectifier(
                          sfFDN::ControllableFullWaveRectifierOptions{.alpha = 1.1f, .sample_rate = kSampleRate}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::ControllableFullWaveRectifier(sfFDN::ControllableFullWaveRectifierOptions{
                          .alpha = 0.5f, .dc_block = true, .sample_rate = 0.f}),
                      std::invalid_argument);

    // The sample rate is irrelevant when the dc blocker is off.
    REQUIRE_NOTHROW(sfFDN::ControllableFullWaveRectifier(
        sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.5f, .dc_block = false, .sample_rate = 0.f}));

    sfFDN::ControllableFullWaveRectifier rectifier;
    REQUIRE_THROWS_AS(rectifier.SetAlpha(2.f), std::invalid_argument);
}

TEST_CASE("ControllableFullWaveRectifier does not allocate")
{
    sfFDN::ControllableFullWaveRectifier rectifier(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 0.6f, .antialiasing = true, .dc_block = true, .sample_rate = kSampleRate});

    std::vector<float> input = MakeNoise(64);
    std::vector<float> output(input.size(), 0.f);
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    rectifier.Process(input_buffer, output_buffer);

    {
        const sfFDNTest::ScopedAllocationCounter counter;
        rectifier.Process(input_buffer, output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

// ==================== SignalDependentFractionalDelay ====================

TEST_CASE("SignalDependentFractionalDelay matches the reference implementation")
{
    const auto d = GENERATE(0.f, 0.25f, 0.5f, 1.f);
    const auto input = MakeNoise(512, 4242);

    sfFDN::SignalDependentFractionalDelay filter(sfFDN::SignalDependentFractionalDelayOptions{.d = d});
    ReferenceSdfd reference(d);

    const auto output = ProcessBlock(filter, input);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(reference(input[i]), 1e-6f));
    }
}

TEST_CASE("SignalDependentFractionalDelay is a plain delay at d = 0")
{
    // At d = 0 both halves are delayed by exactly one sample, so the two branches recombine into x[n - 1].
    const auto input = MakeNoise(128, 5150);
    sfFDN::SignalDependentFractionalDelay filter(sfFDN::SignalDependentFractionalDelayOptions{.d = 0.f});
    const auto output = ProcessBlock(filter, input);

    REQUIRE_THAT(output[0], Catch::Matchers::WithinAbs(0.f, 1e-6f));
    for (auto i = 1u; i < input.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(input[i - 1], 1e-6f));
    }
}

TEST_CASE("SignalDependentFractionalDelay splits the halves at d = 1")
{
    // At d = 1 the negative half is not delayed at all and the positive half is delayed by two samples.
    const auto input = MakeNoise(128, 1234);
    sfFDN::SignalDependentFractionalDelay filter(sfFDN::SignalDependentFractionalDelayOptions{.d = 1.f});
    const auto output = ProcessBlock(filter, input);

    for (auto i = 2u; i < input.size(); ++i)
    {
        const float expected = std::min(input[i], 0.f) + std::max(input[i - 2], 0.f);
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected, 1e-6f));
    }
}

TEST_CASE("SignalDependentFractionalDelay is slightly lossy")
{
    // Section 4.2 of the paper: the overlap between the delayed halves costs up to one sample of energy per period,
    // so the operation loses energy rather than preserving or adding it.
    const auto input = MakeSine(8192, 1760.f / kSampleRate);
    const float input_energy = Energy(input);

    sfFDN::SignalDependentFractionalDelay filter(sfFDN::SignalDependentFractionalDelayOptions{.d = 1.f});
    const auto output = ProcessBlock(filter, input);

    const float ratio = Energy(output) / input_energy;
    REQUIRE(ratio < 1.f);
    REQUIRE(ratio > 0.8f);
}

TEST_CASE("SignalDependentFractionalDelay block processing matches Tick")
{
    const sfFDN::SignalDependentFractionalDelayOptions options{.d = 0.6f};

    sfFDN::SignalDependentFractionalDelay block_processor(options);
    sfFDN::SignalDependentFractionalDelay tick_processor(options);

    const auto input = MakeNoise(256, 2718);
    const auto block_output = ProcessBlock(block_processor, input);

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(block_output[i], Catch::Matchers::WithinAbs(tick_processor.Tick(input[i]), 1e-6f));
    }
}

TEST_CASE("SignalDependentFractionalDelay Clone and Clear")
{
    const sfFDN::SignalDependentFractionalDelayOptions options{.d = 0.3f};
    sfFDN::SignalDependentFractionalDelay filter(options);

    const auto warmup = MakeNoise(32, 11);
    for (const float sample : warmup)
    {
        filter.Tick(sample);
    }

    auto clone = filter.Clone();
    REQUIRE(clone != nullptr);
    REQUIRE(clone->InputChannelCount() == 1);
    REQUIRE(clone->OutputChannelCount() == 1);

    const auto input = MakeNoise(128, 13);
    for (const float sample : input)
    {
        std::array<float, 1> clone_input = {sample};
        std::array<float, 1> clone_output = {0.f};
        sfFDN::AudioBuffer clone_input_buffer(clone_input);
        sfFDN::AudioBuffer clone_output_buffer(clone_output);
        clone->Process(clone_input_buffer, clone_output_buffer);

        REQUIRE_THAT(clone_output[0], Catch::Matchers::WithinAbs(filter.Tick(sample), 1e-6f));
    }

    filter.Clear();
    REQUIRE_THAT(filter.GetD(), Catch::Matchers::WithinAbs(options.d, 1e-6f));

    sfFDN::SignalDependentFractionalDelay fresh(options);
    for (const float sample : input)
    {
        REQUIRE_THAT(filter.Tick(sample), Catch::Matchers::WithinAbs(fresh.Tick(sample), 1e-6f));
    }
}

TEST_CASE("SignalDependentFractionalDelay parameter validation")
{
    REQUIRE_THROWS_AS(sfFDN::SignalDependentFractionalDelay(sfFDN::SignalDependentFractionalDelayOptions{.d = -0.1f}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::SignalDependentFractionalDelay(sfFDN::SignalDependentFractionalDelayOptions{.d = 1.5f}),
                      std::invalid_argument);

    sfFDN::SignalDependentFractionalDelay filter;
    REQUIRE_THROWS_AS(filter.SetD(-1.f), std::invalid_argument);
}

TEST_CASE("SignalDependentFractionalDelay does not allocate")
{
    sfFDN::SignalDependentFractionalDelay filter(sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5f});

    std::vector<float> input = MakeNoise(64);
    std::vector<float> output(input.size(), 0.f);
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    filter.Process(input_buffer, output_buffer);

    {
        const sfFDNTest::ScopedAllocationCounter counter;
        filter.Process(input_buffer, output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

// ==================== RingModulator ====================

TEST_CASE("RingModulator multiplies by the modulating sinusoid")
{
    constexpr float kFrequency = 100.f / kSampleRate;
    const auto input = MakeNoise(1024, 8);

    sfFDN::RingModulator modulator(
        sfFDN::RingModulatorOptions{.frequency = kFrequency, .amplitude = kSqrt2, .initial_phase = 0.f});
    const auto output = ProcessBlock(modulator, input);

    for (auto i = 0u; i < input.size(); ++i)
    {
        const float expected =
            input[i] * kSqrt2 *
            std::sin(2.f * std::numbers::pi_v<float> * kFrequency * static_cast<float>(i));
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected, 1e-3f));
    }
}

TEST_CASE("RingModulator at zero frequency is a constant gain")
{
    // A quarter turn of phase puts the modulator at its peak, where it never moves again.
    const auto input = MakeNoise(64, 77);
    sfFDN::RingModulator modulator(
        sfFDN::RingModulatorOptions{.frequency = 0.f, .amplitude = kSqrt2, .initial_phase = 0.25f});
    const auto output = ProcessBlock(modulator, input);

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(input[i] * kSqrt2, 1e-4f));
    }
}

TEST_CASE("RingModulator is energy preserving on average")
{
    // Section 4.2 of the paper: the average power of a unit sinusoid is one half, so a gain of sqrt(2) restores the
    // energy over a whole modulation period.
    constexpr uint32_t kSize = 96000;
    const auto input = MakeSine(kSize, 1760.f / kSampleRate);

    sfFDN::RingModulator modulator(
        sfFDN::RingModulatorOptions{.frequency = 100.f / kSampleRate, .amplitude = kSqrt2, .initial_phase = 0.f});
    const auto output = ProcessBlock(modulator, input);

    REQUIRE_THAT(Energy(output) / Energy(input), Catch::Matchers::WithinAbs(1.f, 0.05f));
}

TEST_CASE("RingModulator block processing matches Tick")
{
    const sfFDN::RingModulatorOptions options{
        .frequency = 440.f / kSampleRate, .amplitude = kSqrt2, .initial_phase = 0.125f};

    sfFDN::RingModulator block_processor(options);
    sfFDN::RingModulator tick_processor(options);

    const auto input = MakeNoise(256, 424);
    const auto block_output = ProcessBlock(block_processor, input);

    // SineWave::Multiply keeps the phase unwrapped for the length of the block while Tick() wraps it on every sample,
    // so the two accumulate rounding differently. The tolerance covers that, not a difference in behaviour.
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(block_output[i], Catch::Matchers::WithinAbs(tick_processor.Tick(input[i]), 1e-4f));
    }
}

TEST_CASE("RingModulator Clone and Clear")
{
    const sfFDN::RingModulatorOptions options{
        .frequency = 250.f / kSampleRate, .amplitude = kSqrt2, .initial_phase = 0.3f};
    sfFDN::RingModulator modulator(options);

    // Advance the phase so that a clone that restarted it instead of carrying it over would be caught.
    for (auto i = 0u; i < 100u; ++i)
    {
        modulator.Tick(1.f);
    }

    auto clone = modulator.Clone();
    REQUIRE(clone != nullptr);
    REQUIRE(clone->InputChannelCount() == 1);
    REQUIRE(clone->OutputChannelCount() == 1);

    const auto input = MakeNoise(128, 17);
    for (const float sample : input)
    {
        std::array<float, 1> clone_input = {sample};
        std::array<float, 1> clone_output = {0.f};
        sfFDN::AudioBuffer clone_input_buffer(clone_input);
        sfFDN::AudioBuffer clone_output_buffer(clone_output);
        clone->Process(clone_input_buffer, clone_output_buffer);

        REQUIRE_THAT(clone_output[0], Catch::Matchers::WithinAbs(modulator.Tick(sample), 1e-6f));
    }

    modulator.Clear();
    REQUIRE_THAT(modulator.GetFrequency(), Catch::Matchers::WithinAbs(options.frequency, 1e-9f));
    REQUIRE_THAT(modulator.GetAmplitude(), Catch::Matchers::WithinAbs(options.amplitude, 1e-6f));

    sfFDN::RingModulator fresh(options);
    for (const float sample : input)
    {
        REQUIRE_THAT(modulator.Tick(sample), Catch::Matchers::WithinAbs(fresh.Tick(sample), 1e-6f));
    }
}

TEST_CASE("RingModulator parameter validation")
{
    REQUIRE_THROWS_AS(sfFDN::RingModulator(sfFDN::RingModulatorOptions{.frequency = -0.1f}), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::RingModulator(sfFDN::RingModulatorOptions{.initial_phase = 1.5f}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::RingModulator(sfFDN::RingModulatorOptions{.initial_phase = -0.5f}),
                      std::invalid_argument);
}

TEST_CASE("RingModulator does not allocate")
{
    sfFDN::RingModulator modulator(
        sfFDN::RingModulatorOptions{.frequency = 100.f / kSampleRate, .amplitude = kSqrt2, .initial_phase = 0.f});

    std::vector<float> input = MakeNoise(64);
    std::vector<float> output(input.size(), 0.f);
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    modulator.Process(input_buffer, output_buffer);

    {
        const sfFDNTest::ScopedAllocationCounter counter;
        modulator.Process(input_buffer, output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

// ==================== DcBlocker ====================

TEST_CASE("DcBlocker removes a constant offset")
{
    sfFDN::DcBlocker blocker(kSampleRate);

    float last = 0.f;
    for (auto i = 0u; i < 96000u; ++i)
    {
        last = blocker.Tick(1.f);
    }

    REQUIRE(std::abs(last) < 1e-2f);
}

TEST_CASE("DcBlocker make-up gain is bounded")
{
    // The blocker sits inside a feedback loop, so the make-up gain must never run away, even when the input is pure
    // dc and the output is therefore driven to zero.
    sfFDN::DcBlocker blocker(kSampleRate);

    for (auto i = 0u; i < 480000u; ++i)
    {
        const float output = blocker.Tick(1.f);
        REQUIRE(std::abs(output) <= sfFDN::DcBlocker::kMaxGain);
        REQUIRE(std::isfinite(output));
    }
}

TEST_CASE("DcBlocker passes a signal it cannot attenuate")
{
    // Well above the cutoff the blocker is essentially transparent and the make-up gain settles near one.
    const auto input = MakeSine(48000, 1000.f / kSampleRate);
    sfFDN::DcBlocker blocker(kSampleRate);

    std::vector<float> output(input.size(), 0.f);
    for (auto i = 0u; i < input.size(); ++i)
    {
        output[i] = blocker.Tick(input[i]);
    }

    const auto tail = std::span<const float>(output).subspan(output.size() / 2);
    const auto input_tail = std::span<const float>(input).subspan(input.size() / 2);
    REQUIRE_THAT(Energy(tail) / Energy(input_tail), Catch::Matchers::WithinAbs(1.f, 0.05f));
}

// ==================== Multichannel banks ====================

TEST_CASE("Multichannel nonlinearity banks bypass their null channels")
{
    constexpr uint32_t kChannels = 4;
    constexpr uint32_t kBlockSize = 64;

    auto check_bypass = [&](sfFDN::AudioProcessor& bank) {
        REQUIRE(bank.InputChannelCount() == kChannels);
        REQUIRE(bank.OutputChannelCount() == kChannels);

        std::vector<float> input(static_cast<size_t>(kChannels) * kBlockSize, 0.f);
        std::vector<float> output(input.size(), 0.f);

        const auto noise = MakeNoise(kBlockSize, 606);
        for (auto channel = 0u; channel < kChannels; ++channel)
        {
            std::ranges::copy(noise, input.begin() + (static_cast<size_t>(channel) * kBlockSize));
        }

        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannels, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannels, output);
        bank.Process(input_buffer, output_buffer);

        // Channels 0 and 1 are bypassed and must come out untouched, channels 2 and 3 must not.
        for (auto channel = 0u; channel < 2u; ++channel)
        {
            const auto channel_output = output_buffer.GetChannelSpan(channel);
            for (auto i = 0u; i < kBlockSize; ++i)
            {
                REQUIRE_THAT(channel_output[i], Catch::Matchers::WithinAbs(noise[i], 1e-6f));
            }
        }

        for (auto channel = 2u; channel < kChannels; ++channel)
        {
            const auto channel_output = output_buffer.GetChannelSpan(channel);
            bool differs = false;
            for (auto i = 0u; i < kBlockSize; ++i)
            {
                differs = differs || (std::abs(channel_output[i] - noise[i]) > 1e-4f);
            }
            REQUIRE(differs);
        }
    };

    SECTION("ControllableFullWaveRectifier")
    {
        const auto options = sfFDN::MakeMultichannelControllableFullWaveRectifierOptions(1.f, kSampleRate, kChannels, 2);
        REQUIRE(!options.channels[0].has_value());
        REQUIRE(!options.channels[1].has_value());
        REQUIRE(options.channels[2].has_value());
        REQUIRE(options.channels[3].has_value());

        auto bank = sfFDN::MakeMultichannelControllableFullWaveRectifier(options);
        check_bypass(*bank);
    }

    SECTION("SignalDependentFractionalDelay")
    {
        const auto options = sfFDN::MakeMultichannelSignalDependentFractionalDelayOptions(1.f, kChannels, 2);
        auto bank = sfFDN::MakeMultichannelSignalDependentFractionalDelay(options);
        check_bypass(*bank);
    }

    SECTION("RingModulator")
    {
        const auto options =
            sfFDN::MakeMultichannelRingModulatorOptions(100.f / kSampleRate, kSqrt2, kChannels, 2);
        auto bank = sfFDN::MakeMultichannelRingModulator(options);
        check_bypass(*bank);
    }
}

TEST_CASE("Multichannel nonlinearity bank channels are independent")
{
    constexpr uint32_t kChannels = 4;
    constexpr uint32_t kBlockSize = 32;

    auto bank = sfFDN::MakeMultichannelSignalDependentFractionalDelay(
        sfFDN::MakeMultichannelSignalDependentFractionalDelayOptions(0.5f, kChannels));

    std::vector<float> input(static_cast<size_t>(kChannels) * kBlockSize, 0.f);
    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannels, input);
    input_buffer.GetChannelSpan(1)[0] = 1.f;

    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannels, output);
    bank->Process(input_buffer, output_buffer);

    for (auto channel = 0u; channel < kChannels; ++channel)
    {
        const auto channel_output = output_buffer.GetChannelSpan(channel);
        const float energy = Energy(channel_output);
        if (channel == 1)
        {
            REQUIRE(energy > 0.f);
        }
        else
        {
            REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(0.f, 1e-12f));
        }
    }
}

TEST_CASE("Multichannel ring modulator staggers the initial phases")
{
    constexpr uint32_t kChannels = 4;
    const auto options = sfFDN::MakeMultichannelRingModulatorOptions(100.f / kSampleRate, kSqrt2, kChannels);

    REQUIRE(options.channels.size() == kChannels);
    for (auto channel = 0u; channel < kChannels; ++channel)
    {
        REQUIRE(options.channels[channel].has_value());
        REQUIRE_THAT(options.channels[channel]->initial_phase,
                     Catch::Matchers::WithinAbs(static_cast<float>(channel) / static_cast<float>(kChannels), 1e-6f));
    }
}

TEST_CASE("Multichannel nonlinearity banks reject invalid options")
{
    sfFDN::MultichannelControllableFullWaveRectifierOptions rectifier_options;
    rectifier_options.channels.emplace_back(
        sfFDN::ControllableFullWaveRectifierOptions{.alpha = 2.f, .sample_rate = kSampleRate});
    REQUIRE_THROWS_AS(sfFDN::MakeMultichannelControllableFullWaveRectifier(rectifier_options),
                      std::invalid_argument);

    sfFDN::MultichannelSignalDependentFractionalDelayOptions sdfd_options;
    sdfd_options.channels.emplace_back(sfFDN::SignalDependentFractionalDelayOptions{.d = -1.f});
    REQUIRE_THROWS_AS(sfFDN::MakeMultichannelSignalDependentFractionalDelay(sdfd_options), std::invalid_argument);

    sfFDN::MultichannelRingModulatorOptions ring_mod_options;
    ring_mod_options.channels.emplace_back(sfFDN::RingModulatorOptions{.frequency = -1.f});
    REQUIRE_THROWS_AS(sfFDN::MakeMultichannelRingModulator(ring_mod_options), std::invalid_argument);
}

// ==================== In the feedback loop of an FDN ====================

namespace
{
/** @brief Builds the network of Section 5 of the paper: N = 8, a random orthogonal feedback matrix, b = c = 1/N, the
 * eight delay lengths given in the paper, and a first-order attenuation filter with T60(0) = 2 s and T60(pi) = 0.5 s.
 */
sfFDN::FDNConfig MakeShimmerConfig()
{
    constexpr uint32_t kOrder = 8;
    constexpr uint32_t kBlockSize = 64;

    // Section 5 of the paper, in milliseconds.
    constexpr std::array<float, kOrder> kDelaysMs = {71.f, 111.f, 235.f, 297.f, 307.f, 347.f, 381.f, 400.f};

    sfFDN::FDNConfig config;
    config.fdn_size = kOrder;
    config.transposed = false;
    config.direct_gain = 0.f;
    config.block_size = kBlockSize;
    config.sample_rate = kSampleRate;

    std::vector<float> delays(kOrder, 0.f);
    for (auto i = 0u; i < kOrder; ++i)
    {
        delays[i] = std::round(kDelaysMs[i] * kSampleRate / 1000.f);
    }

    config.delay_bank_config = {
        .delays = delays,
        .block_size = kBlockSize,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };

    const float gain = 1.f / static_cast<float>(kOrder);
    config.input_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Split,
        .gains = std::vector<float>(kOrder, gain),
        .time_varying_config = {},
    };
    config.output_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Merge,
        .gains = std::vector<float>(kOrder, gain),
        .time_varying_config = {},
    };

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = kOrder,
        .type = sfFDN::ScalarMatrixType::Random,
        .custom_matrix = std::nullopt,
        .rng_seed = 4242,
        .arg = std::nullopt,
    };

    sfFDN::AttenuationFilterBankOptions attenuation;
    for (auto i = 0u; i < kOrder; ++i)
    {
        attenuation.filter_configs.emplace_back(sfFDN::TwoBandFilterOptions{
            .t60s = {2.f, 0.5f},
            .delay = delays[i],
            .sample_rate = kSampleRate,
        });
    }
    config.attenuation_filter_bank_config = attenuation;

    return config;
}

/** @brief Renders a short burst into the FDN and lets it decay, returning the whole response. */
std::vector<float> RenderDecay(sfFDN::FDN& fdn, uint32_t block_size, uint32_t block_count)
{
    std::vector<float> response;
    response.reserve(static_cast<size_t>(block_size) * block_count);

    std::vector<float> input(block_size, 0.f);
    std::vector<float> output(block_size, 0.f);

    for (auto block = 0u; block < block_count; ++block)
    {
        std::ranges::fill(input, 0.f);
        if (block == 0)
        {
            input[0] = 1.f;
        }

        // FDN::Process accumulates into its output buffer, so it has to be cleared on every block.
        std::ranges::fill(output, 0.f);

        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        fdn.Process(input_buffer, output_buffer);

        response.insert(response.end(), output.begin(), output.end());
    }

    return response;
}

float PeakOf(std::span<const float> signal)
{
    float peak = 0.f;
    for (const float sample : signal)
    {
        peak = std::max(peak, std::abs(sample));
    }
    return peak;
}
} // namespace

TEST_CASE("Nonlinear FDN stays bounded and decays")
{
    constexpr uint32_t kBlockSize = 64;
    // Four seconds at 96 kHz, comfortably longer than the 2 s T60 of the network.
    constexpr uint32_t kBlockCount = 6000;
    constexpr uint32_t kOrder = 8;

    auto run = [&](const sfFDN::multi_channel_processor_variant_t& nonlinearity) {
        auto config = MakeShimmerConfig();
        config.loop_filter_configs.push_back(nonlinearity);

        auto fdn = sfFDN::CreateFDNFromConfig(config);
        REQUIRE(fdn != nullptr);

        const auto response = RenderDecay(*fdn, kBlockSize, kBlockCount);

        for (const float sample : response)
        {
            REQUIRE(std::isfinite(sample));
        }

        const std::span<const float> whole(response);
        const float head_peak = PeakOf(whole.subspan(0, whole.size() / 4));
        const float tail_peak = PeakOf(whole.subspan(3 * whole.size() / 4));

        // Nothing may run away, and the response must still be decaying by the end.
        REQUIRE(PeakOf(whole) < 4.f);
        REQUIRE(tail_peak < head_peak);
    };

    SECTION("ControllableFullWaveRectifier")
    {
        run(sfFDN::MakeMultichannelControllableFullWaveRectifierOptions(1.f, kSampleRate, kOrder));
    }

    SECTION("SignalDependentFractionalDelay")
    {
        run(sfFDN::MakeMultichannelSignalDependentFractionalDelayOptions(1.f, kOrder));
    }

    SECTION("RingModulator")
    {
        run(sfFDN::MakeMultichannelRingModulatorOptions(100.f / kSampleRate, kSqrt2, kOrder));
    }
}

TEST_CASE("Nonlinear FDN generates harmonics that the linear network does not")
{
    // Section 5.1 of the paper: the nonlinearity produces even harmonics of the input, and the recursion fills in the
    // odd ones. Drive the network with a sinusoid and measure the energy of the second harmonic.
    constexpr uint32_t kBlockSize = 64;
    constexpr uint32_t kBlockCount = 1500;
    constexpr uint32_t kOrder = 8;
    constexpr float kFrequency = 1760.f / kSampleRate;

    auto second_harmonic_energy = [&](const std::optional<sfFDN::multi_channel_processor_variant_t>& nonlinearity) {
        auto config = MakeShimmerConfig();
        if (nonlinearity.has_value())
        {
            config.loop_filter_configs.push_back(nonlinearity.value());
        }

        auto fdn = sfFDN::CreateFDNFromConfig(config);

        const uint32_t total = kBlockSize * kBlockCount;
        std::vector<float> response;
        response.reserve(total);

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);

        for (auto block = 0u; block < kBlockCount; ++block)
        {
            for (auto i = 0u; i < kBlockSize; ++i)
            {
                const auto n = static_cast<float>((block * kBlockSize) + i);
                input[i] = 0.5f * std::sin(2.f * std::numbers::pi_v<float> * kFrequency * n);
            }

            sfFDN::AudioBuffer input_buffer(input);
            std::ranges::fill(output, 0.f);
            sfFDN::AudioBuffer output_buffer(output);
            fdn->Process(input_buffer, output_buffer);
            response.insert(response.end(), output.begin(), output.end());
        }

        // Correlate the second half of the response with the second harmonic.
        const std::span<const float> tail = std::span<const float>(response).subspan(response.size() / 2);
        float real = 0.f;
        float imag = 0.f;
        for (auto i = 0u; i < tail.size(); ++i)
        {
            const float phase = 2.f * std::numbers::pi_v<float> * 2.f * kFrequency * static_cast<float>(i);
            real += tail[i] * std::cos(phase);
            imag += tail[i] * std::sin(phase);
        }
        const auto count = static_cast<float>(tail.size());
        return ((real * real) + (imag * imag)) / (count * count);
    };

    const float linear = second_harmonic_energy(std::nullopt);
    const float nonlinear =
        second_harmonic_energy(sfFDN::MakeMultichannelControllableFullWaveRectifierOptions(1.f, kSampleRate, kOrder));

    REQUIRE(nonlinear > 10.f * linear);
}
