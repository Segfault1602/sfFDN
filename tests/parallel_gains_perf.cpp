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
    ModeInfo{sfFDN::ParallelGainsMode::Split, "Split"},
    ModeInfo{sfFDN::ParallelGainsMode::Merge, "Merge"},
    ModeInfo{sfFDN::ParallelGainsMode::Parallel, "Parallel"},
};

void RunParallelGainsBenchmark(const ModeInfo& mode, uint32_t channel_count, uint32_t block_size,
                               nanobench::Bench& bench)
{
    const uint32_t input_channels = mode.mode == sfFDN::ParallelGainsMode::Split ? 1U : channel_count;
    const uint32_t output_channels = mode.mode == sfFDN::ParallelGainsMode::Merge ? 1U : channel_count;
    std::vector<float> input(static_cast<size_t>(input_channels) * block_size);
    std::vector<float> output(static_cast<size_t>(output_channels) * block_size);
    sfFDN::test::perf::FillNoise(input);

    const std::vector<float> gains(channel_count, 0.5F);
    sfFDN::ParallelGains processor(mode.mode, gains);
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

TEST_CASE("ParallelGainsPerf", "[parallel_gains]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "ParallelGains perf");

    for (const ModeInfo& mode : kModes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            for (const uint32_t channel_count : sfFDN::test::perf::kChannelCounts)
            {
                bench.minEpochIterations(9'000'000U / channel_count);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
                RunParallelGainsBenchmark(mode, channel_count, block_size, bench);
            }
        }
    }
}
