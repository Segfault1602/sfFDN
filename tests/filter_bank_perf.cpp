#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
std::unique_ptr<sfFDN::FilterBank> MakeFilterBank(uint32_t channel_count)
{
    auto bank = std::make_unique<sfFDN::FilterBank>();
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        bank->AddFilter(std::make_unique<sfFDN::OnePoleFilter>(0.7F, -0.3F));
    }
    return bank;
}

void RunFilterBankBenchmark(uint32_t channel_count, uint32_t block_size, nanobench::Bench& bench)
{
    auto bank = MakeFilterBank(channel_count);
    std::vector<float> input(static_cast<size_t>(channel_count) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(block_size, channel_count, input);
    sfFDN::AudioBuffer output_buffer(block_size, channel_count, output);

    bench.run("N=" + std::to_string(channel_count) + " B=" + std::to_string(block_size), [&] {
        bank->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("FilterBankPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "FilterBank perf");
    for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
    {
        for (const uint32_t channel_count : sfFDN::test::perf::kChannelCounts)
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
            RunFilterBankBenchmark(channel_count, block_size, bench);
        }
    }
}

TEST_CASE("FilterBankPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "FilterBank B=128");
    for (const uint32_t channel_count : sfFDN::test::perf::kChannelCounts)
    {
        bench.complexityN(channel_count);
        RunFilterBankBenchmark(channel_count, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
