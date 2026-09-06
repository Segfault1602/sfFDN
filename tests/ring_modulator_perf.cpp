#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <chrono>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

using namespace ankerl;

TEST_CASE("RingModulatorPerf", "[nonlinear]")
{
    constexpr auto kSampleRate = static_cast<float>(sfFDN::kDefaultSampleRate);

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "RingModulator perf", std::chrono::milliseconds(200), 2000);

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);

        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        sfFDN::RingModulator processor({
            .frequency = 100.F / kSampleRate,
            .amplitude = std::numbers::sqrt2_v<float>,
            .initial_phase = 0.F,
        });

        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
        bench.run("Process B=" + std::to_string(block_size), [&] {
            processor.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
