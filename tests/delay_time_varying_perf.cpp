#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
struct InterpolationInfo
{
    sfFDN::DelayInterpolationType type;
    std::string_view name;
};

constexpr std::array kInterpolationTypes = {
    InterpolationInfo{sfFDN::DelayInterpolationType::None, "None"},
    InterpolationInfo{sfFDN::DelayInterpolationType::Linear, "Linear"},
    InterpolationInfo{sfFDN::DelayInterpolationType::Allpass, "Allpass"},
    InterpolationInfo{sfFDN::DelayInterpolationType::Lagrange, "Lagrange"},
};

sfFDN::DelayOptions MakeOptions(sfFDN::DelayInterpolationType interpolation)
{
    return {
        .delay = 2048.F,
        .max_delay = 4096U,
        .interp_type = interpolation,
        .lfo_config =
            sfFDN::ModulationOptions{
                .frequency = 1.F / static_cast<float>(sfFDN::kDefaultSampleRate),
                .amplitude = 16.F,
                .initial_phase = 0.125F,
            },
    };
}
} // namespace

TEST_CASE("DelayTimeVaryingPerf", "[delay]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DelayTimeVarying perf");

    for (const InterpolationInfo& interpolation : kInterpolationTypes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            std::vector<float> input(block_size);
            std::vector<float> block_output(block_size);
            std::vector<float> tick_output(block_size);
            sfFDN::test::perf::FillNoise(input);

            sfFDN::DelayTimeVarying block_delay(MakeOptions(interpolation.type));
            sfFDN::DelayTimeVarying tick_delay(MakeOptions(interpolation.type));
            const sfFDN::AudioBuffer input_buffer(input);
            sfFDN::AudioBuffer output_buffer(block_output);
            const std::string suffix =
                " " + std::string(interpolation.name) + " B=" + std::to_string(block_size);

            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            bench.run("Process" + suffix, [&] {
                block_delay.Process(input_buffer, output_buffer);
                nanobench::doNotOptimizeAway(block_output);
            });
            bench.run("Tick" + suffix, [&] {
                for (uint32_t sample = 0; sample < block_size; ++sample)
                {
                    tick_output[sample] = tick_delay.Tick(input[sample]);
                }
                nanobench::doNotOptimizeAway(tick_output);
            });
        }
    }
}
