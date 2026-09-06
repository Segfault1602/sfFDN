#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using namespace ankerl;

TEST_CASE("AllpassFilterPerf", "[filter]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "AllpassFilter perf", std::chrono::milliseconds(20), 100'000);

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        std::vector<float> input(block_size);
        std::vector<float> block_output(block_size);
        std::vector<float> tick_output(block_size);
        sfFDN::test::perf::FillNoise(input);

        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer block_output_buffer(block_output);
        sfFDN::AllpassFilter block_filter({.coeff = 0.5F});
        sfFDN::AllpassFilter tick_filter({.coeff = 0.5F});

        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
        const std::string suffix = " B=" + std::to_string(block_size);

        bench.run("Process" + suffix, [&] {
            block_filter.Process(input_buffer, block_output_buffer);
            nanobench::doNotOptimizeAway(block_output);
        });

        bench.run("Tick" + suffix, [&] {
            for (uint32_t sample = 0; sample < block_size; ++sample)
            {
                tick_output[sample] = tick_filter.Tick(input[sample]);
            }
            nanobench::doNotOptimizeAway(tick_output);
        });
    }
}
