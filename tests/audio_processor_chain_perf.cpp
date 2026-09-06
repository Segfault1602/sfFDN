#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
constexpr std::array kStageCounts = {1U, 2U, 4U, 8U};

std::unique_ptr<sfFDN::AudioProcessorChain> MakeChain(uint32_t block_size, uint32_t stage_count)
{
    auto chain = std::make_unique<sfFDN::AudioProcessorChain>(block_size);
    for (uint32_t stage = 0; stage < stage_count; ++stage)
    {
        REQUIRE(chain->AddProcessor(std::make_unique<sfFDN::OnePoleFilter>(0.7F, -0.3F)));
    }
    return chain;
}

void RunAudioProcessorChainBenchmark(uint32_t stage_count, uint32_t block_size, nanobench::Bench& bench)
{
    auto chain = MakeChain(block_size, stage_count);
    std::vector<float> input(block_size);
    std::vector<float> output(block_size);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    chain->Process(input_buffer, output_buffer);
    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    chain->Clear();

    bench.run("stages=" + std::to_string(stage_count) + " B=" + std::to_string(block_size), [&] {
        chain->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("AudioProcessorChainPerf", "[processor_chain]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "AudioProcessorChain perf");

    for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
    {
        for (const uint32_t stage_count : kStageCounts)
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            RunAudioProcessorChainBenchmark(stage_count, block_size, bench);
        }
    }
}

TEST_CASE("AudioProcessorChainPerf_BigO", "[processor_chain][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "AudioProcessorChain B=128");

    for (const uint32_t stage_count : kStageCounts)
    {
        bench.complexityN(stage_count);
        RunAudioProcessorChainBenchmark(stage_count, kBlockSize, bench);
    }
    std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
}
