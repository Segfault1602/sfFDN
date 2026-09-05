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
} // namespace

TEST_CASE("DelayInterpPerf", "[delay]")
{
    constexpr float kDelay = 2456.5F;
    constexpr uint32_t kMaxDelay = 8192U;

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DelayInterp perf");

    for (const InterpolationInfo& interpolation : kInterpolationTypes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            std::vector<float> input(block_size);
            std::vector<float> block_output(block_size);
            std::vector<float> tick_output(block_size);
            sfFDN::test::perf::FillNoise(input);

            const sfFDN::DelayOptions options{
                .delay = interpolation.type == sfFDN::DelayInterpolationType::None ? 2456.F : kDelay,
                .max_delay = kMaxDelay,
                .interp_type = interpolation.type,
            };
            sfFDN::DelayInterp block_delay(options);
            sfFDN::DelayInterp tick_delay(options);
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
