#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

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
