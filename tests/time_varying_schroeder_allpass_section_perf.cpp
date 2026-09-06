#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
constexpr std::array kStageCounts = {1U, 2U, 4U, 8U};

sfFDN::TimeVaryingSchroederAllpassSectionOptions MakeOptions(uint32_t stage_count, bool parallel)
{
    sfFDN::TimeVaryingSchroederAllpassSectionOptions options;
    options.parallel = parallel;
    options.delays.resize(stage_count);
    options.gains.assign(stage_count, 0.55F);
    options.time_varying_config.resize(stage_count);
    for (uint32_t stage = 0; stage < stage_count; ++stage)
    {
        options.delays[stage] = static_cast<float>(127U + (stage * 74U));
        options.time_varying_config[stage] = {
            .frequency = (0.5F + (0.05F * static_cast<float>(stage))) /
                         static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 0.1F,
            .initial_phase = static_cast<float>(stage) / static_cast<float>(stage_count),
        };
    }
    return options;
}

void RunTimeVaryingSchroederBenchmark(uint32_t stage_count, bool parallel, uint32_t block_size,
                                      nanobench::Bench& bench)
{
    sfFDN::TimeVaryingSchroederAllpassSection processor(MakeOptions(stage_count, parallel));
    std::vector<float> input(block_size);
    std::vector<float> output(block_size);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    const std::string mode = parallel ? "Parallel" : "Serial";

    bench.run(mode + " stages=" + std::to_string(stage_count) + " B=" + std::to_string(block_size), [&] {
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("TimeVaryingSchroederAllpassSectionPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "TimeVaryingSchroederAllpassSection perf");
    for (const bool parallel : {false, true})
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            for (const uint32_t stage_count : kStageCounts)
            {
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
                RunTimeVaryingSchroederBenchmark(stage_count, parallel, block_size, bench);
            }
        }
    }
}

TEST_CASE("TimeVaryingSchroederAllpassSectionPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    for (const bool parallel : {false, true})
    {
        nanobench::Bench bench;
        const std::string mode = parallel ? "Parallel" : "Serial";
        sfFDN::test::perf::ConfigureComplexityBench(bench, "TimeVaryingSchroederAllpassSection " + mode);
        for (const uint32_t stage_count : kStageCounts)
        {
            bench.complexityN(stage_count);
            RunTimeVaryingSchroederBenchmark(stage_count, parallel, kBlockSize, bench);
        }
        std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
    }
}
