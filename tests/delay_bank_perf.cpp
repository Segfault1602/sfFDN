#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
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
    InterpolationInfo{.type = sfFDN::DelayInterpolationType::None, .name = "None"},
    InterpolationInfo{.type = sfFDN::DelayInterpolationType::Linear, .name = "Linear"},
    InterpolationInfo{.type = sfFDN::DelayInterpolationType::Allpass, .name = "Allpass"},
    InterpolationInfo{.type = sfFDN::DelayInterpolationType::Lagrange, .name = "Lagrange"},
};

std::vector<float> MakeDelays(uint32_t channel_count, sfFDN::DelayInterpolationType interpolation)
{
    const float fractional = interpolation == sfFDN::DelayInterpolationType::None ? 0.F : 0.5F;
    std::vector<float> delays(channel_count);
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        delays[channel] = 1024.F + static_cast<float>(73U * channel) + fractional;
    }
    return delays;
}

void RunDelayBankBenchmark(const InterpolationInfo& interpolation, uint32_t channel_count, uint32_t block_size,
                           nanobench::Bench& bench)
{
    sfFDN::DelayBank processor({
        .delays = MakeDelays(channel_count, interpolation.type),
        .block_size = block_size,
        .interpolation_type = interpolation.type,
    });
    std::vector<float> input(static_cast<size_t>(channel_count) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(block_size, channel_count, input);
    sfFDN::AudioBuffer output_buffer(block_size, channel_count, output);
    const std::string name =
        std::string(interpolation.name) + " N=" + std::to_string(channel_count) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("DelayBankPerf", "[delay]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DelayBank perf");

    bool first_benchmark = true;
    for (const InterpolationInfo& interpolation : kInterpolationTypes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t channel_count : sfFDN::test::perf::kChannelCounts)
            {
                bench.warmup(first_benchmark ? 1'000'000 : 100);
                bench.minEpochTime(interpolation.type == sfFDN::DelayInterpolationType::Allpass
                                       ? std::chrono::milliseconds(20)
                                       : std::chrono::milliseconds(10));
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
                RunDelayBankBenchmark(interpolation, channel_count, block_size, bench);
                first_benchmark = false;
            }
        }
    }
}
