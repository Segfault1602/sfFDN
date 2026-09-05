#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
struct RectifierVariant
{
    bool antialiasing;
    bool dc_block;
    std::string_view name;
};

constexpr std::array kVariants = {
    RectifierVariant{false, false, "plain"},
    RectifierVariant{false, true, "plain + dc blocker"},
    RectifierVariant{true, false, "antialiasing"},
    RectifierVariant{true, true, "antialiasing + dc blocker"},
};
} // namespace

TEST_CASE("ControllableFullWaveRectifierPerf", "[nonlinear]")
{
    constexpr float kSampleRate = static_cast<float>(sfFDN::kDefaultSampleRate);

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(
        bench, "ControllableFullWaveRectifier perf", std::chrono::milliseconds(200), 2000);

    for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
    {
        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);
        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);

        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
        for (const RectifierVariant& variant : kVariants)
        {
            sfFDN::ControllableFullWaveRectifier processor({
                .alpha = 1.F,
                .antialiasing = variant.antialiasing,
                .dc_block = variant.dc_block,
                .sample_rate = kSampleRate,
            });
            const std::string name = std::string(variant.name) + " B=" + std::to_string(block_size);

            bench.run(name, [&] {
                processor.Process(input_buffer, output_buffer);
                nanobench::doNotOptimizeAway(output);
            });
        }
    }
}
