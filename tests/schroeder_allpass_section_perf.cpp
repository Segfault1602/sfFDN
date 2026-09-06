#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
constexpr std::array kStageCounts = {1U, 2U, 4U, 8U};

sfFDN::SchroederAllpassSectionOptions MakeOptions(uint32_t stage_count, bool parallel)
{
    sfFDN::SchroederAllpassSectionOptions options;
    options.parallel = parallel;
    options.delays.resize(stage_count);
    options.gains.assign(stage_count, 0.55F);
    for (uint32_t stage = 0; stage < stage_count; ++stage)
    {
        options.delays[stage] = static_cast<float>(127U + (stage * 74U));
    }
    return options;
}

void RunSchroederAllpassSectionBenchmark(uint32_t stage_count, bool parallel, uint32_t block_size,
                                         nanobench::Bench& bench)
{
    sfFDN::SchroederAllpassSection processor(MakeOptions(stage_count, parallel));
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

TEST_CASE("SchroederAllpassSectionPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "SchroederAllpassSection perf");
    for (const bool parallel : {false, true})
    {
        sfFDN::test::perf::SetMinEpochTime(
            bench, parallel ? std::chrono::milliseconds(10) : std::chrono::milliseconds(50));
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t stage_count : kStageCounts)
            {
                sfFDN::test::perf::SetMinEpochIterations(bench, 900'000U / stage_count);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
                RunSchroederAllpassSectionBenchmark(stage_count, parallel, block_size, bench);
            }
        }
    }
}

TEST_CASE("SchroederAllpassSectionPerf_Aliased", "[filter]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "SchroederAllpassSection aliased perf");
    for (const uint32_t stage_count : kStageCounts)
    {
        sfFDN::SchroederAllpassSection processor(MakeOptions(stage_count, true));
        std::vector<float> input(kBlockSize);
        std::vector<float> inout(kBlockSize);
        sfFDN::test::perf::FillNoise(input);
        sfFDN::AudioBuffer buffer(inout);
        sfFDN::test::perf::SetChannelSampleBatch(bench, kBlockSize);
        bench.run("Parallel with input restore stages=" + std::to_string(stage_count), [&] {
            std::ranges::copy(input, inout.begin());
            processor.Process(buffer, buffer);
            nanobench::doNotOptimizeAway(inout);
        });
        REQUIRE(std::ranges::all_of(inout, [](float sample) { return std::isfinite(sample); }));
    }
}

TEST_CASE("SchroederAllpassSectionPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    for (const bool parallel : {false, true})
    {
        nanobench::Bench bench;
        const std::string mode = parallel ? "Parallel" : "Serial";
        sfFDN::test::perf::ConfigureComplexityBench(bench, "SchroederAllpassSection " + mode);
        for (const uint32_t stage_count : kStageCounts)
        {
            bench.complexityN(stage_count);
            RunSchroederAllpassSectionBenchmark(stage_count, parallel, kBlockSize, bench);
        }
        std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
    }
}
