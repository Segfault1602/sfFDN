#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("AudioProcessorChainComparisonPerf", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr std::array<sfFDN::FilterCoefficients, 2> kCoefficients = {{
        {.b0 = 0.75F, .b1 = -0.25F, .b2 = 0.1F, .a0 = 1.F, .a1 = -0.4F, .a2 = 0.2F},
        {.b0 = 0.6F, .b1 = 0.15F, .b2 = -0.05F, .a0 = 1.F, .a1 = -0.3F, .a2 = 0.1F},
    }};

    auto chain = std::make_unique<sfFDN::AudioProcessorChain>(kBlockSize);
    REQUIRE(chain->AddProcessor(std::make_unique<sfFDN::OnePoleFilter>(0.7F, -0.3F)));
    REQUIRE(chain->AddProcessor(std::make_unique<sfFDN::AllpassFilter>(
        sfFDN::AllpassFilterOptions{.coeff = 0.5F})));
    auto chain_cascade = std::make_unique<sfFDN::CascadedBiquads>();
    chain_cascade->SetCoefficients(kCoefficients);
    REQUIRE(chain->AddProcessor(std::move(chain_cascade)));

    sfFDN::OnePoleFilter one_pole(0.7F, -0.3F);
    sfFDN::AllpassFilter allpass({.coeff = 0.5F});
    sfFDN::CascadedBiquads cascade;
    cascade.SetCoefficients(kCoefficients);

    std::vector<float> input(kBlockSize);
    std::vector<float> scratch_a(kBlockSize);
    std::vector<float> scratch_b(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
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

TEST_CASE("DattorroDelayVsTimeVaryingPerf", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 128U;
    const sfFDN::DelayOptions delay_options{
        .delay = 480.F,
        .max_delay = 1024U,
        .interp_type = sfFDN::DelayInterpolationType::Allpass,
        .lfo_config =
            sfFDN::ModulationOptions{
                .frequency = 0.15F / static_cast<float>(sfFDN::kDefaultSampleRate),
                .amplitude = 240.F,
                .initial_phase = 0.F,
            },
    };

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    nanobench::Bench bench;
    bench.title("DattorroDelay vs DelayTimeVarying");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);
    bench.relative(true);

    sfFDN::DelayTimeVarying time_varying(delay_options);
    bench.run("DelayTimeVarying", [&] {
        time_varying.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::DattorroDelay feedforward_only({
        .delay_config = delay_options,
        .blend = 0.7071F,
        .feedforward = 0.7071F,
        .feedback = 0.F,
    });
    bench.run("DattorroDelay (no feedback)", [&] {
        feedforward_only.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::DattorroDelay with_feedback({
        .delay_config = delay_options,
        .blend = 0.7071F,
        .feedforward = 1.F,
        .feedback = 0.7071F,
    });
    bench.run("DattorroDelay (feedback)", [&] {
        with_feedback.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("SchroederAllpassComparisonPerf", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kDelay = 479U;
    constexpr float kGain = 0.55F;

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::test::perf::FillNoise(input);

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
        sfFDN::ModulationOptions{
            .frequency = 0.7F / static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 0.3F,
            .initial_phase = 0.125F,
        });
    bench.run("TimeVaryingSchroederAllpass", [&] {
        modulated_allpass.ProcessBlock(input, output);
        nanobench::doNotOptimizeAway(output);
    });
}
