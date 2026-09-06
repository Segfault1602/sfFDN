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
constexpr std::array kExtendedChannelCounts = {4U, 5U, 6U, 7U, 8U, 12U, 15U, 16U, 17U, 20U, 24U, 28U, 31U, 32U, 33U};
constexpr std::array kStageCounts = {1U, 2U, 4U, 8U, 10U};

std::span<const uint32_t> ChannelCounts()
{
    static constexpr std::array<uint32_t, 1> kDefaultChannelCounts = {8U};
    return sfFDN::test::perf::EnvironmentFlagEnabled("SFFDN_PERF_BLOCK_SWEEP")
               ? std::span<const uint32_t>(kExtendedChannelCounts)
               : std::span<const uint32_t>(kDefaultChannelCounts);
}

std::vector<sfFDN::FilterCoefficients> MakeCoefficients(uint32_t channel_count, uint32_t stage_count)
{
    std::vector<sfFDN::FilterCoefficients> coefficients;
    coefficients.reserve(static_cast<size_t>(channel_count) * stage_count);
    const auto source = std::span(k_h001_AbsorbtionSOS[0]).first(stage_count);
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        coefficients.insert(coefficients.end(), source.begin(), source.end());
    }
    return coefficients;
}

void RunIIRFilterBankBenchmark(uint32_t channel_count, uint32_t stage_count, uint32_t block_size,
                               nanobench::Bench& bench)
{
    sfFDN::IIRFilterBank bank;
    bank.SetFilter(MakeCoefficients(channel_count, stage_count), channel_count);
    std::vector<float> input(static_cast<size_t>(channel_count) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(block_size, channel_count, input);
    sfFDN::AudioBuffer output_buffer(block_size, channel_count, output);

    bench.run("N=" + std::to_string(channel_count) + " stages=" + std::to_string(stage_count) +
                  " B=" + std::to_string(block_size),
              [&] {
                  bank.Process(input_buffer, output_buffer);
                  nanobench::doNotOptimizeAway(output);
              });
}
} // namespace

TEST_CASE("IIRFilterBankPerf", "[filter]")
{
    constexpr uint32_t kStageCount = 10U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "IIRFilterBank channel perf");
    bool first_benchmark = true;
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t channel_count : ChannelCounts())
        {
            sfFDN::test::perf::SetWarmup(bench, first_benchmark ? 100'000U : 100U);
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, channel_count);
            RunIIRFilterBankBenchmark(channel_count, kStageCount, block_size, bench);
            first_benchmark = false;
        }
    }
}

TEST_CASE("IIRFilterBankPerf_Stages", "[filter]")
{
    constexpr uint32_t kChannelCount = 16U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "IIRFilterBank stage perf");
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t stage_count : kStageCounts)
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, kChannelCount);
            RunIIRFilterBankBenchmark(kChannelCount, stage_count, block_size, bench);
        }
    }
}

TEST_CASE("IIRFilterBankPerf_BigO", "[filter][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kStageCount = 10U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "IIRFilterBank B=128 stages=10");
    for (const uint32_t channel_count : kExtendedChannelCounts)
    {
        bench.complexityN(channel_count);
        RunIIRFilterBankBenchmark(channel_count, kStageCount, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
