#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
std::vector<uint32_t> MakeDelays(uint32_t order)
{
    std::vector<uint32_t> delays(static_cast<size_t>(order) * order);
    for (uint32_t destination = 0; destination < order; ++destination)
    {
        for (uint32_t source = 0; source < order; ++source)
        {
            delays[(destination * order) + source] = 16U + ((destination * 17U + source * 31U) % 64U);
        }
    }
    return delays;
}

void RunDelayMatrixBenchmark(uint32_t order, uint32_t block_size, nanobench::Bench& bench)
{
    const std::vector<uint32_t> delays = MakeDelays(order);
    const sfFDN::ScalarFeedbackMatrix mixing_matrix({
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = order,
                .generator = sfFDN::ScalarMatrixType::Random,
                .rng_seed = 4242U,
            },
    });
    sfFDN::DelayMatrix matrix(order, delays, mixing_matrix);

    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);

    bench.run("N=" + std::to_string(order) + " B=" + std::to_string(block_size), [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("DelayMatrixPerf", "[feedback_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DelayMatrix perf");

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t order : sfFDN::test::perf::ChannelCounts())
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
            RunDelayMatrixBenchmark(order, block_size, bench);
        }
    }
}

TEST_CASE("DelayMatrixPerf_BigO", "[feedback_matrix][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "DelayMatrix B=128");

    for (const uint32_t order : sfFDN::test::perf::kExtendedChannelCounts)
    {
        bench.complexityN(order);
        RunDelayMatrixBenchmark(order, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
