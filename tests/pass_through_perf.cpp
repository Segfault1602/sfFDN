#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "passthrough.h"
#include "processor_perf_utils.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using namespace ankerl;

TEST_CASE("PassThroughPerf", "[processor_chain]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "PassThrough perf", std::chrono::milliseconds(20), 100'000);

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);

        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        sfFDN::PassThrough processor;

        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
        bench.run("Process B=" + std::to_string(block_size), [&] {
            processor.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
