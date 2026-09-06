#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "filter_coeffs.h"
#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
constexpr std::array kStageCounts = {1U, 2U, 4U, 8U, 11U};

void RunCascadedBiquadsBenchmark(uint32_t stage_count, uint32_t block_size, nanobench::Bench& bench)
{
    sfFDN::CascadedBiquads processor;
    processor.SetCoefficients(std::span(k_h001_AbsorbtionSOS[0]).first(stage_count));
    std::vector<float> input(block_size);
    std::vector<float> output(block_size);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    bench.run("stages=" + std::to_string(stage_count) + " B=" + std::to_string(block_size), [&] {
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("CascadedBiquadsPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "CascadedBiquads perf");
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t stage_count : kStageCounts)
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            RunCascadedBiquadsBenchmark(stage_count, block_size, bench);
        }
    }
}

TEST_CASE("CascadedBiquadsPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "CascadedBiquads B=128");
    for (const uint32_t stage_count : kStageCounts)
    {
        bench.complexityN(stage_count);
        RunCascadedBiquadsBenchmark(stage_count, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
