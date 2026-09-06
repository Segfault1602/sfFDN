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
constexpr std::array kTapCounts = {16U, 32U, 64U, 128U, 256U};

void RunFirBenchmark(uint32_t tap_count, uint32_t block_size, nanobench::Bench& bench)
{
    std::vector<float> coefficients(tap_count);
    sfFDN::test::perf::FillNoise(coefficients, 0xA511E9B3U + tap_count);
    sfFDN::Fir processor({.coeffs = coefficients});
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

TEST_CASE("FirPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "Fir perf");
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t tap_count : kTapCounts)
        {
            bench.minEpochIterations(tap_count == 64U && block_size == 128U ? 200'000U : 1U);
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            RunFirBenchmark(tap_count, block_size, bench);
        }
    }
}

TEST_CASE("FirPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "Fir B=128");
    for (const uint32_t tap_count : kTapCounts)
    {
        bench.complexityN(tap_count);
        RunFirBenchmark(tap_count, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
