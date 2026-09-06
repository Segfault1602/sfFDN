#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
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

sfFDN::DelayBankTimeVaryingOptions MakeOptions(uint32_t channel_count, sfFDN::DelayInterpolationType interpolation)
{
    sfFDN::DelayBankTimeVaryingOptions options;
    options.max_delay = 8192U;
    options.interpolation_type = interpolation;
    options.delays.resize(channel_count);
    options.time_varying_config.resize(channel_count);
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        options.delays[channel] = 1024.F + static_cast<float>(73U * channel);
        options.time_varying_config[channel] = {
            .frequency = (0.5F + (0.05F * static_cast<float>(channel))) / static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 16.F,
            .initial_phase = static_cast<float>(channel) / static_cast<float>(channel_count),
        };
    }
    return options;
}

void RunDelayBankTimeVaryingBenchmark(const InterpolationInfo& interpolation, uint32_t channel_count,
                                      uint32_t block_size, nanobench::Bench& bench)
{
    sfFDN::DelayBankTimeVarying processor(MakeOptions(channel_count, interpolation.type));
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

TEST_CASE("DelayBankTimeVaryingPerf", "[delay]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DelayBankTimeVarying perf");

    for (const InterpolationInfo& interpolation : kInterpolationTypes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t channel_count : sfFDN::test::perf::kChannelCounts)
            {
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
                RunDelayBankTimeVaryingBenchmark(interpolation, channel_count, block_size, bench);
            }
        }
    }
}
