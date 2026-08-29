#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sffdn/audio_buffer.h"
#include "sffdn/delay_interp.h"
#include "sffdn/delay_utils.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "rng.h"
#include "test_utils.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <ranges>
#include <sndfile.h>
#include <span>
#include <vector>

namespace
{
void TestDelayBlock(float delay, uint32_t block_size, uint32_t max_delay, sfFDN::DelayInterpolationType interp_type)
{
    INFO("delay: " << delay << " block_size: " << block_size
                   << " interp: " << static_cast<int>(interp_type)); // NOLINT
    constexpr uint32_t kBlockCount = 4;

    sfFDN::DelayOptions config{delay, max_delay, interp_type};
    sfFDN::DelayInterp delay_sample(config);
    sfFDN::DelayInterp delay_block(config);

    std::vector<float> input(static_cast<size_t>(block_size) * kBlockCount, 0.f);
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i);
    }

    std::vector<float> output_sample(block_size, 0.f);
    std::vector<float> output_block(block_size, 0.f);

    for (uint32_t block = 0; block < kBlockCount; ++block)
    {
        const std::span<float> input_block(input.data() + (static_cast<size_t>(block) * block_size), block_size);

        for (uint32_t i = 0; i < block_size; ++i)
        {
            output_sample[i] = delay_sample.Tick(input_block[i]);
        }

        sfFDN::AudioBuffer input_buffer(block_size, 1, input_block);
        sfFDN::AudioBuffer output_buffer(block_size, 1, output_block);
        delay_block.Process(input_buffer, output_buffer);

        for (auto [out, expected] : std::views::zip(output_block, output_sample))
        {
            REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 1e-4));
        }
    }
}

// Exercises the read-before-write half of the block API, which is the order the FDN feedback loop uses: a block is
// read out of the delay bank, mixed, and only then written back in.
void TestDelayBlockReadBeforeWrite(float delay, uint32_t block_size, uint32_t max_delay,
                                   sfFDN::DelayInterpolationType interp_type)
{
    INFO("delay: " << delay << " block_size: " << block_size
                   << " interp: " << static_cast<int>(interp_type)); // NOLINT
    constexpr uint32_t kBlockCount = 4;

    const sfFDN::DelayOptions config{delay, max_delay, interp_type};

    std::vector<float> input(static_cast<size_t>(block_size) * kBlockCount, 0.f);
    sfFDN::RNG rng;
    for (float& sample : input)
    {
        sample = rng();
    }

    sfFDN::DelayInterp delay_sample(config);
    sfFDN::DelayInterp delay_block(config);

    std::vector<float> output_sample(block_size, 0.f);
    std::vector<float> output_block(block_size, 0.f);

    for (uint32_t block = 0; block < kBlockCount; ++block)
    {
        const std::span<const float> input_block(input.data() + (static_cast<size_t>(block) * block_size), block_size);

        for (uint32_t i = 0; i < block_size; ++i)
        {
            output_sample[i] = delay_sample.NextOut();
            delay_sample.Advance(input_block[i]);
        }

        std::ranges::fill(output_block, 0.f);
        delay_block.GetNextOutputs(output_block);
        REQUIRE(delay_block.AddNextInputs(input_block));

        for (auto [out, expected] : std::views::zip(output_block, output_sample))
        {
            REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 1e-5));
        }
    }
}

// Crest factor of the second difference of a signal. A discontinuity shows up as an isolated large second
// difference, so a smooth signal has a crest factor of a few units and a clicking one of tens.
float SecondDifferenceCrestFactor(std::span<const float> signal)
{
    float peak = 0.f;
    float sum_of_squares = 0.f;
    uint32_t count = 0;
    for (size_t n = 2; n < signal.size(); ++n)
    {
        const float d2 = signal[n] - (2.f * signal[n - 1]) + signal[n - 2];
        peak = std::max(peak, std::abs(d2));
        sum_of_squares += d2 * d2;
        ++count;
    }

    const float rms = std::sqrt(sum_of_squares / static_cast<float>(count));
    return rms > 0.f ? peak / rms : 0.f;
}
} // namespace

TEST_CASE("Delay")
{
    sfFDN::Delay delay(1, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    // for (uint32_t i = 0; i < iteration; ++i)
    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }

    // Test copy ctor
    sfFDN::Delay delay_copy(delay);
    REQUIRE(delay_copy.GetDelay() == delay.GetDelay());
}

std::vector<float> ProcessDelay(float delay, uint32_t max_delay, uint32_t block_size,
                                sfFDN::DelayInterpolationType interp_type, std::span<float> input)
{
    sfFDN::DelayInterp delay_block({delay, max_delay, interp_type});

    std::vector<float> output_block(block_size, 0.f);

    sfFDN::AudioBuffer input_buffer(block_size, 1, input);
    sfFDN::AudioBuffer output_buffer(block_size, 1, output_block);

    delay_block.Process(input_buffer, output_buffer);

    return output_block;
}

TEST_CASE("Delay_Integer")
{
    constexpr uint32_t kBlockSize = 64;
    constexpr uint32_t kMaxDelay = 128;
    constexpr float kDelay = 5.f;
    std::vector<float> input;

    sfFDN::RNG rng;
    for (uint32_t i = 0; i < kBlockSize; ++i)
    {
        input.push_back(rng());
    }

    auto output_none = ProcessDelay(kDelay, kMaxDelay, kBlockSize, sfFDN::DelayInterpolationType::None, input);
    auto output_linear = ProcessDelay(kDelay, kMaxDelay, kBlockSize, sfFDN::DelayInterpolationType::Linear, input);
    auto output_allpass = ProcessDelay(kDelay, kMaxDelay, kBlockSize, sfFDN::DelayInterpolationType::Allpass, input);
    auto output_lagrange = ProcessDelay(kDelay, kMaxDelay, kBlockSize, sfFDN::DelayInterpolationType::Lagrange, input);

    for (uint32_t i = 0; i < kBlockSize; ++i)
    {
        REQUIRE_THAT(output_none[i],
                     Catch::Matchers::WithinAbs(output_linear[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_linear[i],
                     Catch::Matchers::WithinAbs(output_allpass[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_allpass[i],
                     Catch::Matchers::WithinAbs(output_lagrange[i], std::numeric_limits<float>::epsilon()));

        std::cout << "Input: " << input[i] << ", None: " << output_none[i] << ", Linear: " << output_linear[i]
                  << ", Allpass: " << output_allpass[i] << ", Lagrange: " << output_lagrange[i] << std::endl;
    }
}

TEST_CASE("DelayTapOut")
{
    sfFDN::Delay delay(8, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        delay.Tick(i);
        output.push_back(delay.TapOut(1));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayMultiTap")
{
    sfFDN::Delay delay(8, 64);

    constexpr uint32_t kBlockSize = 32;
    std::vector<float> input_block(kBlockSize, 0.f);
    input_block[0] = 1.f;

    delay.AddNextInputs(input_block);

    std::vector<uint32_t> taps = {0, 2, 4, 6, 8, 16};
    std::vector<float> coeffs(taps.size(), 1.f);
    std::vector<float> output_block(kBlockSize, 0.f);
    delay.GetNextOutputsAt(taps, output_block, coeffs);

    for (auto i : output_block)
    {
        std::cout << i << ", ";
    }
}

TEST_CASE("ZeroDelay")
{
    sfFDN::Delay delay(0, 10);

    constexpr uint32_t kIteration = 10;
    std::vector<float> output;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));

        REQUIRE(output[i] == delay.TapOut(0));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayA")
{
    sfFDN::DelayInterp delay({1.5, 10, sfFDN::DelayInterpolationType::Allpass});

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }

    sfFDN::DelayInterp delay_block({1.5f, 32, sfFDN::DelayInterpolationType::Allpass});

    std::vector<float> input_block(kIteration, 0.f);
    for (auto i = 0u; i < input_block.size(); ++i)
    {
        input_block[i] = i;
    }

    std::vector<float> output_block(kIteration, 0.f);

    sfFDN::AudioBuffer input_buffer(kIteration, 1, input_block);
    sfFDN::AudioBuffer output_buffer(kIteration, 1, output_block);

    delay_block.Process(input_buffer, output_buffer);

    for (auto [out, expected] : std::views::zip(output_block, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }
}

TEST_CASE("DelayA_MinDelay")
{
    sfFDN::DelayInterp delay({0.5, 10, sfFDN::DelayInterpolationType::Allpass});

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50, 8.50};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }
}

TEST_CASE("DelayLagrange")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kInputSize = 128;
    sfFDN::DelayInterp delay({1.5f, kInputSize * 2, sfFDN::DelayInterpolationType::Lagrange});

    std::vector<float> input(kInputSize, 0.f);
    sfFDN::SineWave sine(200.f / kSampleRate, 0.f);
    sine.SetAmplitude(0.5f);
    sine.Generate(input);
    std::vector<float> output(kInputSize, 0.f);

    // for (uint32_t i = 0; i < input.size(); ++i)
    // {
    //     output[i] = delay.Tick(input[i]);
    // }

    sfFDN::AudioBuffer input_buffer(kInputSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kInputSize, 1, output);
    delay.Process(input_buffer, output_buffer);

    WriteWavFile("delay_lagrange_input.wav", input);
    WriteWavFile("delay_lagrange_output.wav", output);
}

TEST_CASE("DelayBlock")
{
    constexpr uint32_t kBlockSize = 32;
    constexpr uint32_t kMaxDelay = 64;
    TestDelayBlock(5, kBlockSize, kMaxDelay, sfFDN::DelayInterpolationType::None);
    TestDelayBlock(5.34f, kBlockSize, kMaxDelay, sfFDN::DelayInterpolationType::Linear);
    TestDelayBlock(5.34f, kBlockSize, kMaxDelay, sfFDN::DelayInterpolationType::Allpass);
    TestDelayBlock(5.34f, kBlockSize, kMaxDelay, sfFDN::DelayInterpolationType::Lagrange);
    TestDelayBlock(20.34f, 21, 40, sfFDN::DelayInterpolationType::Lagrange);
    TestDelayBlock(20.34f, 22, 40, sfFDN::DelayInterpolationType::Lagrange);
    TestDelayBlock(20.34f, kBlockSize, 40, sfFDN::DelayInterpolationType::Lagrange);
}

TEST_CASE("DelayInterp NextOut and Advance match Tick")
{
    constexpr uint32_t kSampleCount = 64;
    constexpr std::array<sfFDN::DelayInterpolationType, 4> kInterpTypes = {
        sfFDN::DelayInterpolationType::None, sfFDN::DelayInterpolationType::Linear,
        sfFDN::DelayInterpolationType::Allpass, sfFDN::DelayInterpolationType::Lagrange};
    constexpr std::array<float, 3> kDelays = {2.f, 7.25f, 20.34f};

    std::array<float, kSampleCount> input{};
    sfFDN::RNG rng;
    for (float& sample : input)
    {
        sample = rng();
    }

    for (const auto interp_type : kInterpTypes)
    {
        for (const float delay : kDelays)
        {
            const sfFDN::DelayOptions config{.delay = delay, .max_delay = 64, .interp_type = interp_type};

            sfFDN::DelayInterp tick_delay(config);
            sfFDN::DelayInterp split_delay(config);

            for (const float sample : input)
            {
                const float expected = tick_delay.Tick(sample);

                // NextOut() is cached, so calling it twice must return the same value.
                const float actual = split_delay.NextOut();
                REQUIRE(split_delay.NextOut() == actual);
                split_delay.Advance(sample);

                REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 1e-6f));
            }
        }
    }
}

TEST_CASE("DelayInterp TapOut")
{
    constexpr uint32_t kSampleCount = 16;
    const sfFDN::DelayOptions config{
        .delay = 4.f, .max_delay = 64, .interp_type = sfFDN::DelayInterpolationType::Allpass};

    sfFDN::DelayInterp delay(config);

    std::array<float, kSampleCount> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i) + 1.f;
    }

    for (auto n = 0u; n < input.size(); ++n)
    {
        delay.Advance(input[n]);

        // After writing input[n], TapOut(tap) returns input[n - tap], independently of the current delay and of the
        // interpolation type.
        for (auto tap = 0u; tap <= n; ++tap)
        {
            REQUIRE(delay.TapOut(tap) == input[n - tap]);
        }
    }

    // The tap is unaffected by a change of the current delay.
    delay.SetDelay(9.7f);
    REQUIRE(delay.TapOut(0) == input.back());
    REQUIRE(delay.TapOut(3) == input[input.size() - 4]);
}

TEST_CASE("DelayBlockReadBeforeWrite")
{
    // The FDN reads a whole block out of each delay line before writing the feedback block back in, so every
    // interpolation type must agree with the per-sample NextOut()/Advance() pair in that order too.
    constexpr uint32_t kMaxDelay = 128;
    for (const auto interp_type :
         {sfFDN::DelayInterpolationType::None, sfFDN::DelayInterpolationType::Linear,
          sfFDN::DelayInterpolationType::Allpass, sfFDN::DelayInterpolationType::Lagrange})
    {
        TestDelayBlockReadBeforeWrite(40.f, 32, kMaxDelay, interp_type);
        TestDelayBlockReadBeforeWrite(40.34f, 32, kMaxDelay, interp_type);
        TestDelayBlockReadBeforeWrite(66.75f, 64, kMaxDelay, interp_type);
        TestDelayBlockReadBeforeWrite(37.5f, 21, kMaxDelay, interp_type);
        TestDelayBlockReadBeforeWrite(33.01f, 32, kMaxDelay, interp_type);
    }
}

TEST_CASE("DelayInterp allpass tap crossing")
{
    // Sweeping the delay across integer sample boundaries used to leave the allpass filter primed with a sample
    // taken from the tap it had just left, which injected a step into the output on every crossing.
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kSampleCount = 8192;
    constexpr uint32_t kSkip = 512; // let the delay line fill before measuring
    constexpr float kStartDelay = 20.6f;
    constexpr float kEndDelay = 12.4f; // eight integer crossings
    constexpr float kFrequency = 440.f;

    std::vector<float> input(kSampleCount, 0.f);
    for (uint32_t n = 0; n < kSampleCount; ++n)
    {
        input[n] = 0.5f * std::sin(2.f * std::numbers::pi_v<float> * kFrequency * static_cast<float>(n) /
                                   static_cast<float>(kSampleRate));
    }

    auto sweep = [&](sfFDN::DelayInterpolationType interp_type) {
        sfFDN::DelayInterp delay({kStartDelay, 64, interp_type});
        std::vector<float> output(kSampleCount, 0.f);
        for (uint32_t n = 0; n < kSampleCount; ++n)
        {
            const float t = static_cast<float>(n) / static_cast<float>(kSampleCount - 1);
            delay.SetDelay(kStartDelay + ((kEndDelay - kStartDelay) * t));
            output[n] = delay.Tick(input[n]);
        }
        return output;
    };

    const auto linear = sweep(sfFDN::DelayInterpolationType::Linear);
    const auto allpass = sweep(sfFDN::DelayInterpolationType::Allpass);

    const float linear_crest = SecondDifferenceCrestFactor(std::span(linear).subspan(kSkip));
    const float allpass_crest = SecondDifferenceCrestFactor(std::span(allpass).subspan(kSkip));

    INFO("linear crest: " << linear_crest << " allpass crest: " << allpass_crest); // NOLINT

    // A clean sweep has a crest factor of roughly 2; the stale-state bug pushed the allpass path above 20.
    REQUIRE(allpass_crest < 4.f);
    REQUIRE(allpass_crest < 2.f * linear_crest);
}

TEST_CASE("DelayInterp minimum delay")
{
    // Delays smaller than what the structure can represent used to underflow int_delay_ (an uint32_t) and leave the
    // delay line pointing at an out of range tap. They are now clamped to the smallest representable delay.
    constexpr uint32_t kImpulseLength = 64;

    // The centre of gravity of the impulse response is the group delay at DC, which is the delay the structure
    // actually realises.
    auto effective_delay = [](sfFDN::DelayInterp& delay) {
        float sum = 0.f;
        float weighted_sum = 0.f;
        for (uint32_t n = 0; n < kImpulseLength; ++n)
        {
            const float out = delay.Tick(n == 0 ? 1.f : 0.f);
            REQUIRE(std::isfinite(out));
            sum += out;
            weighted_sum += out * static_cast<float>(n);
        }
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.f, 1e-4f));
        return weighted_sum / sum;
    };

    for (const auto interp_type : {sfFDN::DelayInterpolationType::None, sfFDN::DelayInterpolationType::Linear,
                                   sfFDN::DelayInterpolationType::Allpass, sfFDN::DelayInterpolationType::Lagrange})
    {
        for (const float requested : {0.f, 0.1f, 0.25f, 0.49f, 0.5f, 0.51f, 0.99f, 1.f, 1.01f, 1.5f, 2.f, 2.75f})
        {
            INFO("interp: " << static_cast<int>(interp_type) << " requested: " << requested); // NOLINT

            sfFDN::DelayInterp delay({4.f, 64, interp_type});
            delay.SetDelay(requested);
            delay.Clear();

            REQUIRE(delay.GetDelay() == requested);

            float expected = requested;
            switch (interp_type)
            {
            case sfFDN::DelayInterpolationType::None:
                expected = std::floor(requested);
                break;
            case sfFDN::DelayInterpolationType::Allpass:
                expected = std::max(requested, 0.5f);
                break;
            case sfFDN::DelayInterpolationType::Lagrange:
                expected = std::max(requested, 1.f);
                break;
            default:
                break;
            }

            REQUIRE_THAT(effective_delay(delay), Catch::Matchers::WithinAbs(expected, 1e-3f));
        }
    }
}

TEST_CASE("DelayTimeVarying block processing matches Tick"){
    constexpr uint32_t kBlockSize = 32;
    const sfFDN::DelayOptions config{
        .delay = 20.f,
        .max_delay = 64,
        .interp_type = sfFDN::DelayInterpolationType::Linear,
        .lfo_config = sfFDN::ModulationOptions{.frequency = 0.003f, .amplitude = 2.f, .initial_phase = 0.125f},
    };

    std::array<float, kBlockSize> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i) / static_cast<float>(input.size());
    }

    sfFDN::DelayTimeVarying tick_delay(config);
    std::array<float, kBlockSize> expected{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        expected[i] = tick_delay.Tick(input[i]);
    }

    sfFDN::DelayTimeVarying block_delay(config);
    std::array<float, kBlockSize> output{};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    block_delay.Process(input_buffer, output_buffer);

    for (const auto [actual, expected_sample] : std::views::zip(output, expected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected_sample, 1e-5f));
    }
}

TEST_CASE("DelayBank")
{
    constexpr uint32_t kNumDelay = 4;
    const std::vector<float> kDelays = {2, 3, 4, 5};
    sfFDN::DelayBank delay_bank({kDelays, 10});

    std::vector<float> output;

    std::array<float, kNumDelay> impulse = {1, 1, 1, 1};
    std::array<float, 4> buffer = {0, 0, 0, 0};

    sfFDN::AudioBuffer impulse_buffer(1, kNumDelay, impulse);
    sfFDN::AudioBuffer buffer_audio(1, kNumDelay, buffer);

    delay_bank.Process(impulse_buffer, buffer_audio);
    output.reserve(buffer.size());
    for (auto& i : buffer)
    {
        output.push_back(i);
    }

    constexpr uint32_t kIter = 9;
    for (uint32_t i = 0; i < kIter; ++i)
    {
        delay_bank.GetNextOutputs(buffer_audio);
        for (auto& i : buffer)
        {
            output.push_back(i);
        }

        buffer.fill(0);
        delay_bank.AddNextInputs(buffer_audio);
    }

    constexpr std::array<float, 10> kDelay0Expected = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay1Expected = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay2Expected = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay3Expected = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    REQUIRE(output.size() == 40);
    for (uint32_t i = 0; i < output.size(); i += 4)
    {
        REQUIRE_THAT(output[i],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 1],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 2],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 3],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankTimeVarying")
{
    constexpr uint32_t kNumDelay = 4;
    constexpr uint32_t kBlockSize = 8;

    const sfFDN::DelayBankTimeVaryingOptions config{.delays = {2, 3, 4, 5},
                                                    .max_delay = 16,
                                                    .interpolation_type = sfFDN::DelayInterpolationType::Linear,
                                                    .time_varying_config = {}};

    sfFDN::DelayBankTimeVarying delay_bank(config);

    std::vector<float> input(kNumDelay * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (uint32_t i = 0; i < kNumDelay; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kNumDelay * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kNumDelay, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kNumDelay, output);

    delay_bank.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> kDelay0Expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay1Expected = {0, 0, 0, 1, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay2Expected = {0, 0, 0, 0, 1, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay3Expected = {0, 0, 0, 0, 0, 1, 0, 0};

    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output_buffer.GetChannelSpan(0)[j],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(1)[j],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(2)[j],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(3)[j],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(j), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankProcess")
{
    constexpr uint32_t kBlockSize = 8;
    constexpr uint32_t kNumDelay = 4;
    const std::vector<float> kDelays = {0.f, 1.f, 2.f, 3.f};
    sfFDN::DelayBank delay_bank({kDelays, kBlockSize});

    std::vector<float> output;

    std::array<float, kNumDelay * kBlockSize> impulse = {0.f};
    for (auto i = 0u; i < kNumDelay; ++i)
    {
        impulse.at(i * kBlockSize) = 1.f;
    }
    std::array<float, kNumDelay * kBlockSize> buffer = {0.f};

    sfFDN::AudioBuffer impulse_buffer(kBlockSize, kNumDelay, impulse);
    sfFDN::AudioBuffer buffer_audio(kBlockSize, kNumDelay, buffer);

    delay_bank.Process(impulse_buffer, buffer_audio);

    constexpr std::array<float, kBlockSize> kDelay0Expected = {1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay1Expected = {0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay2Expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay3Expected = {0, 0, 0, 1, 0, 0, 0, 0};

    for (uint32_t i = 0; i < kBlockSize; ++i)
    {
        REQUIRE_THAT(buffer_audio.GetChannelSpan(0)[i],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(1)[i],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(2)[i],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(3)[i],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(i), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("Delay block processing preserves wrap and remainder samples")
{
    constexpr uint32_t kDelay = 11;
    constexpr uint32_t kMaximumDelay = 19;
    constexpr std::array<uint32_t, 4> kBlockSizes = {7, 7, 3, 5};

    sfFDN::Delay tick_delay(kDelay, kMaximumDelay);
    sfFDN::Delay block_delay(kDelay, kMaximumDelay);

    uint32_t sample_index = 0;
    for (uint32_t iteration = 0; iteration < 12; ++iteration)
    {
        const uint32_t block_size = kBlockSizes[iteration % kBlockSizes.size()];
        std::vector<float> input(block_size);
        std::vector<float> expected(block_size);
        std::vector<float> output(block_size);

        for (uint32_t sample = 0; sample < block_size; ++sample)
        {
            input[sample] = static_cast<float>(++sample_index);
            expected[sample] = tick_delay.Tick(input[sample]);
        }

        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        block_delay.Process(input_buffer, output_buffer);

        REQUIRE(output == expected);
    }
}

TEST_CASE("Delay split block processing preserves two-segment wraps")
{
    constexpr uint32_t kDelay = 4;
    constexpr uint32_t kMaximumDelay = 7;
    constexpr uint32_t kBlockSize = 3;

    sfFDN::Delay tick_delay(kDelay, kMaximumDelay);
    sfFDN::Delay block_delay(kDelay, kMaximumDelay);

    uint32_t sample_index = 0;
    for (uint32_t iteration = 0; iteration < 10; ++iteration)
    {
        std::array<float, kBlockSize> input{};
        std::array<float, kBlockSize> expected{};
        std::array<float, kBlockSize> output{};
        for (uint32_t sample = 0; sample < kBlockSize; ++sample)
        {
            input[sample] = static_cast<float>(++sample_index);
            expected[sample] = tick_delay.Tick(input[sample]);
        }

        REQUIRE(block_delay.AddNextInputs(input));
        block_delay.GetNextOutputs(output);
        REQUIRE(output == expected);
    }
}

TEST_CASE("Delay GetNextOutputs under-run leaves output and read state unchanged")
{
    sfFDN::Delay delay(3, 7);
    constexpr std::array<float, 4> kInput = {1.f, 2.f, 3.f, 4.f};
    REQUIRE(delay.AddNextInputs(kInput));

    std::array<float, 6> initial_output{};
    delay.GetNextOutputs(initial_output);
    REQUIRE(initial_output == std::array<float, 6>{0.f, 0.f, 0.f, 1.f, 2.f, 3.f});
    REQUIRE(delay.NextOut() == 4.f);
    REQUIRE(delay.LastOut() == 3.f);

    std::array<float, 2> under_run_output = {-1.f, -2.f};
    delay.GetNextOutputs(under_run_output);
    REQUIRE(under_run_output == std::array<float, 2>{-1.f, -2.f});
    REQUIRE(delay.NextOut() == 4.f);
    REQUIRE(delay.LastOut() == 3.f);

    std::array<float, 1> remaining_output{};
    delay.GetNextOutputs(remaining_output);
    REQUIRE(remaining_output == std::array<float, 1>{4.f});
}

TEST_CASE("Delay Process exhaustively matches Tick for small delays and block sizes")
{
    constexpr uint32_t kMaximumDelay = 12;
    constexpr uint32_t kIterationCount = 6;

    for (uint32_t delay = 0; delay <= kMaximumDelay; ++delay)
    {
        for (uint32_t block_size = 1; block_size <= kMaximumDelay + 3; ++block_size)
        {
            sfFDN::Delay tick_delay(delay, kMaximumDelay);
            sfFDN::Delay block_delay(delay, kMaximumDelay);
            uint32_t sample_index = 0;

            for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration)
            {
                CAPTURE(delay, block_size, iteration);
                std::vector<float> input(block_size);
                std::vector<float> expected(block_size);
                std::vector<float> output(block_size);

                for (uint32_t sample = 0; sample < block_size; ++sample)
                {
                    input[sample] = static_cast<float>(++sample_index);
                    expected[sample] = tick_delay.Tick(input[sample]);
                }

                sfFDN::AudioBuffer input_buffer(input);
                sfFDN::AudioBuffer output_buffer(output);
                block_delay.Process(input_buffer, output_buffer);
                REQUIRE(output == expected);
            }
        }
    }
}

TEST_CASE("DelayBank non-native remainder blocks match per-sample delays")
{
    constexpr uint32_t kChannelCount = 3;
    constexpr uint32_t kConfiguredBlockSize = 8;
    constexpr uint32_t kSampleCount = 13;
    constexpr std::array<float, kChannelCount> kDelays = {17.f, 23.f, 31.f};

    sfFDN::DelayBank delay_bank({.delays = {17.f, 23.f, 31.f}, .block_size = kConfiguredBlockSize});
    std::array<sfFDN::DelayInterp, kChannelCount> references = {
        sfFDN::DelayInterp({kDelays[0], 64, sfFDN::DelayInterpolationType::None}),
        sfFDN::DelayInterp({kDelays[1], 64, sfFDN::DelayInterpolationType::None}),
        sfFDN::DelayInterp({kDelays[2], 64, sfFDN::DelayInterpolationType::None}),
    };

    std::array<float, kChannelCount * kSampleCount> input{};
    std::array<float, kChannelCount * kSampleCount> output{};
    uint32_t sample_index = 0;

    for (uint32_t iteration = 0; iteration < 12; ++iteration)
    {
        for (uint32_t channel = 0; channel < kChannelCount; ++channel)
        {
            for (uint32_t sample = 0; sample < kSampleCount; ++sample)
            {
                input[channel * kSampleCount + sample] = static_cast<float>(++sample_index);
            }
        }

        sfFDN::AudioBuffer input_buffer(kSampleCount, kChannelCount, input);
        sfFDN::AudioBuffer output_buffer(kSampleCount, kChannelCount, output);
        delay_bank.Process(input_buffer, output_buffer);

        for (uint32_t channel = 0; channel < kChannelCount; ++channel)
        {
            for (uint32_t sample = 0; sample < kSampleCount; ++sample)
            {
                const float expected = references[channel].Tick(input[channel * kSampleCount + sample]);
                REQUIRE(output[channel * kSampleCount + sample] == expected);
            }
        }
    }
}

TEST_CASE("Delay and DelayBank wrapped steady-state processing does not allocate")
{
    constexpr uint32_t kBlockSize = 7;
    constexpr uint32_t kChannelCount = 3;
    std::array<float, kBlockSize> mono_input{};
    std::array<float, kBlockSize> mono_output{};
    std::array<float, kChannelCount * kBlockSize> bank_input{};
    std::array<float, kChannelCount * kBlockSize> bank_output{};

    sfFDN::Delay delay(11, 19);
    sfFDN::DelayBank delay_bank({.delays = {17.f, 23.f, 31.f}, .block_size = kBlockSize});
    sfFDN::AudioBuffer mono_input_buffer(mono_input);
    sfFDN::AudioBuffer mono_output_buffer(mono_output);
    sfFDN::AudioBuffer bank_input_buffer(kBlockSize, kChannelCount, bank_input);
    sfFDN::AudioBuffer bank_output_buffer(kBlockSize, kChannelCount, bank_output);

    for (uint32_t iteration = 0; iteration < 4; ++iteration)
    {
        delay.Process(mono_input_buffer, mono_output_buffer);
        delay_bank.GetNextOutputs(bank_output_buffer);
        delay_bank.AddNextInputs(bank_input_buffer);
    }

    sfFDNTest::ScopedAllocationCounter allocation_counter;
    for (uint32_t iteration = 0; iteration < 16; ++iteration)
    {
        delay.Process(mono_input_buffer, mono_output_buffer);
        delay_bank.GetNextOutputs(bank_output_buffer);
        delay_bank.AddNextInputs(bank_input_buffer);
    }
    REQUIRE(allocation_counter.Count() == 0);
}

TEST_CASE("DelayInterp_Linear")
{
    sfFDN::DelayInterp delay({1.1f, 10, sfFDN::DelayInterpolationType::Linear});

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }

    sfFDN::DelayInterp delay_block({1.1f, 32, sfFDN::DelayInterpolationType::Linear});

    std::vector<float> input_block(kIteration, 0.f);
    for (auto i = 0u; i < input_block.size(); ++i)
    {
        input_block[i] = i;
    }

    std::vector<float> output_block(kIteration, 0.f);

    sfFDN::AudioBuffer input_buffer(kIteration, 1, input_block);
    sfFDN::AudioBuffer output_buffer(kIteration, 1, output_block);

    delay_block.Process(input_buffer, output_buffer);

    for (auto [out, expected] : std::views::zip(output_block, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }
}

#if 0
TEST_CASE("DelayTimeVarying")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 512;
    constexpr uint32_t kInputSize = kBlockSize * 200;
    constexpr uint32_t kBaseDelay = 1024;

    sfFDN::DelayTimeVarying<sfFDN::DelayInterpolationType::Linear> delay(kBaseDelay, 4096);

    std::vector<float> input(kInputSize, 0.f);
    sfFDN::SineWave sine(200.f / kSampleRate, 0.f);
    sine.SetAmplitude(0.5f);
    sine.Generate(input);
    std::vector<float> output(kInputSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kInputSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kInputSize, 1, output);

    for (auto i = 0u; i < kInputSize; i += kBlockSize)
    {
        sfFDN::AudioBuffer in_block = input_buffer.Offset(i, kBlockSize);
        sfFDN::AudioBuffer out_block = output_buffer.Offset(i, kBlockSize);
        delay.Process(in_block, out_block);
    }

    WriteWavFile("delay_time_varying_input.wav", output);

    output.clear();
    output.resize(kInputSize, 0.f);

    delay.SetMod(1.f / kSampleRate, 256.f);
    delay.Clear();

    // for (auto i = 0u; i < kInputSize; ++i)
    // {
    //     output[i] = delay.Tick(input[i]);
    // }

    for (auto i = 0u; i < kInputSize; i += kBlockSize)
    {
        sfFDN::AudioBuffer in_block = input_buffer.Offset(i, kBlockSize);
        sfFDN::AudioBuffer out_block = output_buffer.Offset(i, kBlockSize);
        delay.Process(in_block, out_block);
    }

    WriteWavFile("delay_time_varying_output.wav", output);
}

TEST_CASE("DelayFeedback")
{

    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 32;
    constexpr uint32_t kInputSize = kBlockSize * 2000;
    constexpr float kBaseDelay = 607.5f;

    sfFDN::DelayTimeVarying<sfFDN::DelayInterpolationType::Linear> delay(kBaseDelay, 4096);
    delay.SetMod(1.f / kSampleRate, 32.f);

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f; // Impulse

    std::vector<float> output(kInputSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kInputSize, 1, output);

    for (auto i = 0u; i < kInputSize; i += kBlockSize)
    {
        sfFDN::AudioBuffer out_block = output_buffer.Offset(i, kBlockSize);
        delay.Process(input_buffer, out_block);
        input_buffer = out_block;
    }

    WriteWavFile("delay_feedback_output.wav", output);
}
#endif