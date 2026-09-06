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
constexpr uint32_t kMaximumTap = 4096U;
constexpr std::array kTapCounts = {4U, 8U, 16U, 32U, 64U};

sfFDN::SparseFirOptions MakeOptions(uint32_t tap_count)
{
    sfFDN::SparseFirOptions options;
    options.coeffs.reserve(tap_count);
    for (uint32_t tap = 0; tap < tap_count; ++tap)
    {
        const uint32_t index = tap * (kMaximumTap / tap_count);
        options.coeffs.emplace_back(index, 1.F / static_cast<float>(tap_count));
    }
    return options;
}

void RunSparseFirBenchmark(uint32_t tap_count, uint32_t block_size, nanobench::Bench& bench)
{
    sfFDN::SparseFir processor(MakeOptions(tap_count));
    std::vector<float> input(block_size);
    std::vector<float> output(block_size);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    bench.run("taps=" + std::to_string(tap_count) + " B=" + std::to_string(block_size), [&] {
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("SparseFirPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "SparseFir perf");
    for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
    {
        for (const uint32_t tap_count : kTapCounts)
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            RunSparseFirBenchmark(tap_count, block_size, bench);
        }
    }
}

TEST_CASE("SparseFirPerf_BigO", "[filter]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "SparseFir B=128");
    for (const uint32_t tap_count : kTapCounts)
    {
        bench.complexityN(tap_count);
        RunSparseFirBenchmark(tap_count, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
