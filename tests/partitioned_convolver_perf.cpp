#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
constexpr std::array kRirLengths = {4096U, 24000U, 96000U};
constexpr std::array kRepCounts = {0U, 8U, 16U};

std::vector<float> MakeRir(uint32_t sample_count)
{
    std::vector<float> rir(sample_count);
    sfFDN::test::perf::FillNoise(rir, 0xA511E9B3U + sample_count);
    for (uint32_t sample = 0; sample < sample_count; ++sample)
    {
        const float decay = std::exp(-6.F * static_cast<float>(sample) / static_cast<float>(sample_count));
        rir[sample] *= decay;
    }
    rir[0] = 1.F;
    return rir;
}
} // namespace

TEST_CASE("PartitionedConvolverPerf", "[convolution]")
{
    const uint32_t loop_count = sfFDN::test::perf::SmokeModeEnabled() ? 8U : 256U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "PartitionedConvolver perf");

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);
        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);

        for (const uint32_t rir_length : kRirLengths)
        {
            const std::vector<float> rir = MakeRir(rir_length);
            for (const uint32_t rep_count : kRepCounts)
            {
                sfFDN::PartitionedConvolver convolver(block_size, rir, rep_count);
                const std::string schedule = rep_count == 0U ? "auto" : "rep=" + std::to_string(rep_count);
                const std::string name = "RIR=" + std::to_string(rir_length) + " " + schedule +
                                         " B=" + std::to_string(block_size);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, 1U, loop_count);

                convolver.Process(input_buffer, output_buffer);
                REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
                convolver.Clear();

                bench.run(name, [&] {
                    for (uint32_t iteration = 0; iteration < loop_count; ++iteration)
                    {
                        convolver.Process(input_buffer, output_buffer);
                    }
                    nanobench::doNotOptimizeAway(output);
                });
            }
        }
    }
}
