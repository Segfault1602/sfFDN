#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{
constexpr uint32_t kChannelCount = 8U;
constexpr uint32_t kBlockSize = 128U;
constexpr float kGain = 0.5F;

/** How the destination is treated before the processor runs.
 *
 * This is the whole subtlety of the comparison. ParallelGains Merge accumulates into its destination, so it only
 * produces a correct standalone block if the destination was cleared first; ChannelMatrix overwrites with its first
 * Scale and needs no clear. But FDN::PrepareOutput clears the wet scratch unconditionally, whichever processor is
 * installed, so inside an FDN both pay for the clear. Reporting only the standalone policy would credit ChannelMatrix
 * with an advantage the FDN does not actually realize.
 */
enum class ClearPolicy : uint8_t
{
    Never,
    Always,
};

std::vector<float> MakeCoefficients(uint32_t input_channels, uint32_t output_channels)
{
    return std::vector<float>(static_cast<size_t>(input_channels) * output_channels, kGain);
}

template <typename Processor>
void RunBoundaryBenchmark(Processor& processor, std::string_view name, uint32_t input_channels,
                          uint32_t output_channels, ClearPolicy clear_policy, nanobench::Bench& bench,
                          std::vector<float>& captured_output)
{
    std::vector<float> input(static_cast<size_t>(input_channels) * kBlockSize);
    std::vector<float> output(static_cast<size_t>(output_channels) * kBlockSize, 0.F);
    sfFDN::test::perf::FillNoise(input);

    const sfFDN::AudioBuffer input_buffer(kBlockSize, input_channels, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, output_channels, output);

    bench.run(std::string(name), [&] {
        if (clear_policy == ClearPolicy::Always)
        {
            std::ranges::fill(output, 0.F);
        }
        processor.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    captured_output = output;
}
} // namespace

TEST_CASE("ChannelMatrixComparisonPerf", "[channel_matrix][.diagnostic]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "ChannelMatrix vs ParallelGains (us per Process call)",
                                                std::chrono::milliseconds(50));
    // These blocks are only ~1 us of work, so a low iteration floor leaves nanobench flagging rows as unstable.
    sfFDN::test::perf::SetMinEpochIterations(bench, 2'500'000U);

    const std::vector<float> gains(kChannelCount, kGain);

    // Split shape, 1 -> N. Both processors overwrite every output channel, so neither needs a cleared destination and
    // the two policies coincide.
    {
        sfFDN::ParallelGains parallel_split(sfFDN::ParallelGainsMode::Split, gains);
        sfFDN::ChannelMatrix matrix_split(sfFDN::ChannelMatrixOptions{
            .input_channel_count = 1U,
            .output_channel_count = kChannelCount,
            .coefficients = MakeCoefficients(1U, kChannelCount),
        });

        std::vector<float> parallel_output;
        std::vector<float> matrix_output;
        RunBoundaryBenchmark(parallel_split, "ParallelGains Split 1->8", 1U, kChannelCount, ClearPolicy::Never, bench,
                             parallel_output);
        RunBoundaryBenchmark(matrix_split, "ChannelMatrix 1->8", 1U, kChannelCount, ClearPolicy::Never, bench,
                             matrix_output);
        // Equivalence is checked outside the timed region so the comparison is known to be like-for-like.
        REQUIRE(parallel_output == matrix_output);
    }

    // Merge shape, N -> 1, as a standalone processor producing a fresh block. ParallelGains must clear first because
    // it only accumulates; ChannelMatrix does not.
    {
        sfFDN::ParallelGains parallel_merge(sfFDN::ParallelGainsMode::Merge, gains);
        sfFDN::ChannelMatrix matrix_merge(sfFDN::ChannelMatrixOptions{
            .input_channel_count = kChannelCount,
            .output_channel_count = 1U,
            .coefficients = MakeCoefficients(kChannelCount, 1U),
        });

        std::vector<float> parallel_output;
        std::vector<float> matrix_output;
        RunBoundaryBenchmark(parallel_merge, "ParallelGains Merge 8->1 standalone (clear+process)", kChannelCount, 1U,
                             ClearPolicy::Always, bench, parallel_output);
        RunBoundaryBenchmark(matrix_merge, "ChannelMatrix 8->1 standalone (process only)", kChannelCount, 1U,
                             ClearPolicy::Never, bench, matrix_output);
        REQUIRE(parallel_output == matrix_output);
    }

    // Merge shape as the FDN actually drives it: PrepareOutput clears the wet scratch either way, so both processors
    // carry the clear. This is the number that answers "what does an FDNConfig boundary matrix cost?".
    {
        sfFDN::ParallelGains parallel_merge(sfFDN::ParallelGainsMode::Merge, gains);
        sfFDN::ChannelMatrix matrix_merge(sfFDN::ChannelMatrixOptions{
            .input_channel_count = kChannelCount,
            .output_channel_count = 1U,
            .coefficients = MakeCoefficients(kChannelCount, 1U),
        });

        std::vector<float> parallel_output;
        std::vector<float> matrix_output;
        RunBoundaryBenchmark(parallel_merge, "ParallelGains Merge 8->1 as FDN drives it", kChannelCount, 1U,
                             ClearPolicy::Always, bench, parallel_output);
        RunBoundaryBenchmark(matrix_merge, "ChannelMatrix 8->1 as FDN drives it", kChannelCount, 1U,
                             ClearPolicy::Always, bench, matrix_output);
        REQUIRE(parallel_output == matrix_output);
    }
}
