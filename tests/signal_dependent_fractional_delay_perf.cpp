#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using namespace ankerl;

TEST_CASE("SignalDependentFractionalDelayPerf", "[nonlinear]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(
        bench, "SignalDependentFractionalDelay perf", std::chrono::milliseconds(200), 2000);

    for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
    {
        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);

        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        sfFDN::SignalDependentFractionalDelay processor({.d = 0.5F});

        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
        bench.run("Process B=" + std::to_string(block_size), [&] {
            processor.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
