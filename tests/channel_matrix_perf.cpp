#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
/** The boundary shapes a MIMO FDN actually uses: the two degenerate stage-gain shapes, a square routing, and the
 * stereo rectangles that are the reason ChannelMatrix exists. */
struct ShapeInfo
{
    uint32_t input_scale;  //!< Input channel count, or 0 to use the sweep's channel count.
    uint32_t output_scale; //!< Output channel count, or 0 to use the sweep's channel count.
};

constexpr std::array kShapes = {
    ShapeInfo{.input_scale = 1U, .output_scale = 0U}, // 1 -> N, the Split stage-gain shape
    ShapeInfo{.input_scale = 0U, .output_scale = 1U}, // N -> 1, the Merge stage-gain shape
    ShapeInfo{.input_scale = 0U, .output_scale = 0U}, // N -> N
    ShapeInfo{.input_scale = 2U, .output_scale = 0U}, // 2 -> N, a stereo input boundary
    ShapeInfo{.input_scale = 0U, .output_scale = 2U}, // N -> 2, a stereo output boundary
};

uint32_t ResolveCount(uint32_t scale, uint32_t channel_count)
{
    return scale == 0U ? channel_count : scale;
}

void RunChannelMatrixBenchmark(uint32_t input_channels, uint32_t output_channels, uint32_t block_size,
                               nanobench::Bench& bench)
{
    std::vector<float> input(static_cast<size_t>(input_channels) * block_size);
    std::vector<float> output(static_cast<size_t>(output_channels) * block_size);
    sfFDN::test::perf::FillNoise(input);

    sfFDN::ChannelMatrix processor(sfFDN::ChannelMatrixOptions{
        .input_channel_count = input_channels,
        .output_channel_count = output_channels,
        .coefficients = std::vector<float>(static_cast<size_t>(input_channels) * output_channels, 0.5F),
    });

    const sfFDN::AudioBuffer input_buffer(block_size, input_channels, input);
    sfFDN::AudioBuffer output_buffer(block_size, output_channels, output);
    const std::string name =
        std::to_string(input_channels) + "->" + std::to_string(output_channels) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("ChannelMatrixPerf", "[channel_matrix]")
{
    nanobench::Bench bench;
    // The effective channel count of a rectangular boundary is max(M, K), matching the existing convention where a
    // dense NxN feedback matrix is normalized by its order rather than by its coefficient count. That makes 1->N and
    // N->1 directly comparable with the corresponding ParallelGains Split and Merge rows.
    sfFDN::test::perf::ConfigureThroughputBench(bench, "ChannelMatrix perf (per effective channel-sample)",
                                                std::chrono::milliseconds(50));

    for (const ShapeInfo& shape : kShapes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t channel_count : sfFDN::test::perf::ChannelCounts())
            {
                const uint32_t input_channels = ResolveCount(shape.input_scale, channel_count);
                const uint32_t output_channels = ResolveCount(shape.output_scale, channel_count);
                // Work per call is proportional to M * K * block_size, so the iteration floor scales inversely with
                // the coefficient count. The cheap 1->N and N->1 shapes are otherwise flagged unstable.
                const uint64_t coefficient_count = static_cast<uint64_t>(input_channels) * output_channels;
                sfFDN::test::perf::SetMinEpochIterations(bench, 20'000'000U / coefficient_count);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, std::max(input_channels, output_channels));
                RunChannelMatrixBenchmark(input_channels, output_channels, block_size, bench);
            }
        }
    }
}
