#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
struct ModeInfo
{
    sfFDN::ParallelGainsMode mode;
    std::string_view name;
};

constexpr std::array kModes = {
    ModeInfo{.mode = sfFDN::ParallelGainsMode::Split, .name = "Split"},
    ModeInfo{.mode = sfFDN::ParallelGainsMode::Merge, .name = "Merge"},
    ModeInfo{.mode = sfFDN::ParallelGainsMode::Parallel, .name = "Parallel"},
};

sfFDN::ParallelGainsOptions MakeOptions(const ModeInfo& mode, uint32_t channel_count)
{
    sfFDN::ParallelGainsOptions options;
    options.mode = mode.mode;
    options.gains.assign(channel_count, 0.5F);
    options.time_varying_config.resize(channel_count);
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        options.time_varying_config[channel] = {
            .frequency = (0.5F + (0.05F * static_cast<float>(channel))) / static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 0.1F,
            .initial_phase = static_cast<float>(channel) / static_cast<float>(channel_count),
        };
    }
    return options;
}

void RunTimeVaryingParallelGainsBenchmark(const ModeInfo& mode, uint32_t channel_count, uint32_t block_size,
                                          nanobench::Bench& bench)
{
    const uint32_t input_channels = mode.mode == sfFDN::ParallelGainsMode::Split ? 1U : channel_count;
    const uint32_t output_channels = mode.mode == sfFDN::ParallelGainsMode::Merge ? 1U : channel_count;
    std::vector<float> input(static_cast<size_t>(input_channels) * block_size);
    std::vector<float> output(static_cast<size_t>(output_channels) * block_size);
    sfFDN::test::perf::FillNoise(input);

    sfFDN::TimeVaryingParallelGains processor(MakeOptions(mode, channel_count));
    const sfFDN::AudioBuffer input_buffer(block_size, input_channels, input);
    sfFDN::AudioBuffer output_buffer(block_size, output_channels, output);
    const std::string name =
        std::string(mode.name) + " N=" + std::to_string(channel_count) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        if (mode.mode == sfFDN::ParallelGainsMode::Merge)
        {
            std::ranges::fill(output, 0.F);
        }
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("TimeVaryingParallelGainsPerf", "[parallel_gains]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "TimeVaryingParallelGains perf");

    for (const ModeInfo& mode : kModes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t channel_count : sfFDN::test::perf::ChannelCounts())
            {
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
                RunTimeVaryingParallelGainsBenchmark(mode, channel_count, block_size, bench);
            }
        }
    }
}
