#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("AudioProcessorChainPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr std::array<sfFDN::FilterCoefficients, 2> kCoefficients = {{
        {.b0 = 0.75f, .b1 = -0.25f, .b2 = 0.1f, .a0 = 1.f, .a1 = -0.4f, .a2 = 0.2f},
        {.b0 = 0.6f, .b1 = 0.15f, .b2 = -0.05f, .a0 = 1.f, .a1 = -0.3f, .a2 = 0.1f},
    }};

    auto chain = std::make_unique<sfFDN::AudioProcessorChain>(kBlockSize);
    chain->AddProcessor(std::make_unique<sfFDN::OnePoleFilter>(0.7f, -0.3f));
    chain->AddProcessor(std::make_unique<sfFDN::AllpassFilter>(sfFDN::AllpassFilterOptions{.coeff = 0.5f}));
    auto chain_cascade = std::make_unique<sfFDN::CascadedBiquads>();
    chain_cascade->SetCoefficients(kCoefficients);
    chain->AddProcessor(std::move(chain_cascade));

    sfFDN::OnePoleFilter one_pole(0.7f, -0.3f);
    sfFDN::AllpassFilter allpass({.coeff = 0.5f});
    sfFDN::CascadedBiquads cascade;
    cascade.SetCoefficients(kCoefficients);

    std::vector<float> input(kBlockSize);
    std::vector<float> scratch_a(kBlockSize);
    std::vector<float> scratch_b(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer scratch_a_buffer(scratch_a);
    sfFDN::AudioBuffer scratch_b_buffer(scratch_b);
    sfFDN::AudioBuffer output_buffer(output);

    nanobench::Bench bench;
    bench.title("AudioProcessorChain overhead");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);
    bench.relative(true);

    bench.run("Direct concrete processors", [&] {
        one_pole.Process(input_buffer, scratch_a_buffer);
        allpass.Process(scratch_a_buffer, scratch_b_buffer);
        cascade.Process(scratch_b_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
    bench.run("AudioProcessorChain", [&] {
        chain->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("DelayBankTimeVaryingPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr std::array<uint32_t, 2> kOrders = {8, 32};

    nanobench::Bench bench;
    bench.title("DelayBankTimeVarying scaling");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);

    for (const uint32_t order : kOrders)
    {
        sfFDN::DelayBankTimeVaryingOptions options;
        options.max_delay = 8192;
        options.interpolation_type = sfFDN::DelayInterpolationType::Linear;
        options.delays.resize(order);
        options.time_varying_config.resize(order);
        for (auto channel = 0u; channel < order; ++channel)
        {
            options.delays[channel] = 1000.f + static_cast<float>(channel * 73u);
            options.time_varying_config[channel] = {
                .frequency = 0.0001f * static_cast<float>(channel + 1),
                .amplitude = 16.f,
                .initial_phase = static_cast<float>(channel) / static_cast<float>(order),
            };
        }

        sfFDN::DelayBankTimeVarying delay_bank(options);
        std::vector<float> input(kBlockSize * order);
        std::vector<float> output(kBlockSize * order);
        sfFDN::RNG generator;
        for (float& sample : input)
        {
            sample = generator();
        }
        sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, order, output);

        bench.run("N=" + std::to_string(order), [&] {
            delay_bank.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("DattorroDelayPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr float kSampleRate = 48000.f;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    nanobench::Bench bench;
    bench.title("DattorroDelay Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);

    constexpr std::array<std::pair<sfFDN::DattorroEffectType, const char*>, 3> kPresets = {{
        {sfFDN::DattorroEffectType::Flanger, "Flanger"},
        {sfFDN::DattorroEffectType::WhiteChorus, "WhiteChorus"},
        {sfFDN::DattorroEffectType::Echo, "Echo"},
    }};

    for (const auto& [type, name] : kPresets)
    {
        sfFDN::DattorroDelay delay(sfFDN::MakeDattorroDelayOptions(type, kSampleRate));
        bench.run(name, [&] {
            delay.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("MultichannelDattorroDelayPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kChannelCount = 8;
    constexpr float kSampleRate = 48000.f;

    // The cost of putting one chorus per channel in an FDN feedback loop.
    auto bank = sfFDN::MakeMultichannelDattorroDelay(sfFDN::MakeMultichannelDattorroDelayOptions(
        sfFDN::DattorroEffectType::WhiteChorus, kSampleRate, kChannelCount));

    std::vector<float> input(static_cast<size_t>(kBlockSize) * kChannelCount, 0.f);
    std::vector<float> output(input.size(), 0.f);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    nanobench::Bench bench;
    bench.title("MultichannelDattorroDelay Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);

    bench.run("WhiteChorus x8", [&] {
        bank->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("DattorroDelayVsTimeVaryingPerf")
{
    constexpr uint32_t kBlockSize = 128;

    // Both processors get the exact same delay line, modulation and interpolation, so the benchmark only measures the
    // cost of the comb filter built around the delay line.
    const sfFDN::DelayOptions kDelayConfig{
        .delay = 480.f,
        .max_delay = 1024,
        .interp_type = sfFDN::DelayInterpolationType::Allpass,
        .lfo_config = sfFDN::ModulationOptions{.frequency = 0.15f / 48000.f, .amplitude = 240.f, .initial_phase = 0.f},
    };

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    nanobench::Bench bench;
    bench.title("DattorroDelay vs DelayTimeVarying");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);
    bench.relative(true);

    sfFDN::DelayTimeVarying time_varying(kDelayConfig);
    bench.run("DelayTimeVarying", [&] {
        time_varying.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    // Without feedback, the extra work over a plain time-varying delay is the fixed tap read and the two gains.
    sfFDN::DattorroDelay feedforward_only(sfFDN::DattorroDelayOptions{
        .delay_config = kDelayConfig,
        .blend = 0.7071f,
        .feedforward = 0.7071f,
        .feedback = 0.f,
    });
    bench.run("DattorroDelay (no feedback)", [&] {
        feedforward_only.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    // With feedback the delay line cannot be read a block at a time, since every input sample depends on the previous
    // output of the delay line.
    sfFDN::DattorroDelay with_feedback(sfFDN::DattorroDelayOptions{
        .delay_config = kDelayConfig,
        .blend = 0.7071f,
        .feedforward = 1.f,
        .feedback = 0.7071f,
    });
    bench.run("DattorroDelay (feedback)", [&] {
        with_feedback.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("SchroederAllpassComparisonPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kDelay = 479;
    constexpr float kGain = 0.55f;
    constexpr float kSampleRate = 48000.f;

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    nanobench::Bench bench;
    bench.title("SchroederAllpass vs TimeVaryingSchroederAllpass");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);
    bench.relative(true);

    sfFDN::SchroederAllpass static_allpass(kDelay, kGain);
    bench.run("SchroederAllpass", [&] {
        static_allpass.ProcessBlock(input, output);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::TimeVaryingSchroederAllpass modulated_allpass(
        kDelay, kGain,
        sfFDN::ModulationOptions{.frequency = 0.7f / kSampleRate, .amplitude = 0.3f, .initial_phase = 0.125f});
    bench.run("TimeVaryingSchroederAllpass (modulated gain)", [&] {
        modulated_allpass.ProcessBlock(input, output);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::DattorroDelay modulated_delay(
        sfFDN::MakeDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate));
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    bench.run("Dattorro modulated delay context", [&] {
        modulated_delay.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
