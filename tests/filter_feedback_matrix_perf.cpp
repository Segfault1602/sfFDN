#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
constexpr uint64_t kMaximumEstimatedDelayStorage = 64ULL * 1024ULL * 1024ULL;

struct Configuration
{
    sfFDN::ScalarMatrixType type;
    float gain_per_samples;
    std::string_view name;
};

constexpr std::array kConfigurations = {
    Configuration{
        .type = sfFDN::ScalarMatrixType::Hadamard,
        .gain_per_samples = 1.F,
        .name = "Hadamard structured",
    },
    Configuration{
        .type = sfFDN::ScalarMatrixType::Householder,
        .gain_per_samples = 1.F,
        .name = "Householder structured",
    },
    Configuration{
        .type = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 1.F,
        .name = "Random dense",
    },
    Configuration{
        .type = sfFDN::ScalarMatrixType::Hadamard,
        .gain_per_samples = 0.9999F,
        .name = "Hadamard with decay",
    },
};

uint64_t EstimateDelayStorage(uint32_t order, uint32_t stage_count, float sparsity)
{
    double pulse_size = 1.0;
    uint64_t bytes = 0;
    for (uint32_t stage = 0; stage < stage_count; ++stage)
    {
        const double stage_sparsity = stage == 0U ? static_cast<double>(sparsity) : 1.0;
        const double maximum_shift = stage_sparsity * static_cast<double>(order) * pulse_size;
        const uint64_t maximum_delay = static_cast<uint64_t>(maximum_shift) + (2U * sfFDN::kDefaultBlockSize);
        const uint64_t rounded_delay = ((maximum_delay + 63U) / 64U) * 64U;
        bytes += static_cast<uint64_t>(order) * (rounded_delay + 1U) * sizeof(float);
        pulse_size *= static_cast<double>(order) * stage_sparsity;
    }
    return bytes;
}

sfFDN::CascadedFeedbackMatrixOptions MakeOptions(const Configuration& configuration, uint32_t order,
                                                 uint32_t stage_count)
{
    constexpr float kSparsity = 1.F;
    REQUIRE(EstimateDelayStorage(order, stage_count, kSparsity) <= kMaximumEstimatedDelayStorage);
    return {
        .matrix_size = order,
        .stage_count = stage_count,
        .sparsity = kSparsity,
        .type = configuration.type,
        .gain_per_samples = configuration.gain_per_samples,
    };
}

void RunFilterFeedbackMatrixBenchmark(const Configuration& configuration, uint32_t order, uint32_t stage_count,
                                      uint32_t block_size, nanobench::Bench& bench)
{
    sfFDN::FilterFeedbackMatrix matrix(MakeOptions(configuration, order, stage_count));
    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);
    const std::string name = std::string(configuration.name) + " N=" + std::to_string(order) +
                             " stages=" + std::to_string(stage_count) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("FilterFeedbackMatrixPerf", "[feedback_matrix]")
{
    constexpr uint32_t kStageCount = 2U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "FilterFeedbackMatrix perf");

    for (const Configuration& configuration : kConfigurations)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t order : sfFDN::test::perf::ChannelCounts())
            {
                const bool uses_structured_kernel = configuration.gain_per_samples == 1.F &&
                                                    (configuration.type == sfFDN::ScalarMatrixType::Hadamard ||
                                                     configuration.type == sfFDN::ScalarMatrixType::Householder);
                const bool needs_iteration_floor =
                    (uses_structured_kernel && order <= 8U && block_size <= 128U) ||
                    (configuration.gain_per_samples != 1.F && order == 64U && block_size == 64U);
                sfFDN::test::perf::SetMinEpochIterations(bench, needs_iteration_floor ? 125'000U : 1U);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
                RunFilterFeedbackMatrixBenchmark(configuration, order, kStageCount, block_size, bench);
            }
        }
    }
}

TEST_CASE("FilterFeedbackMatrixPerf_BigO", "[feedback_matrix][.diagnostic]")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr std::array kStageCounts = {0U, 1U, 2U, 3U, 4U};

    for (const Configuration& configuration : kConfigurations)
    {
        nanobench::Bench bench;
        sfFDN::test::perf::ConfigureComplexityBench(bench, "FilterFeedbackMatrix " + std::string(configuration.name) +
                                                               " N=" + std::to_string(kOrder));

        for (const uint32_t stage_count : kStageCounts)
        {
            bench.complexityN(stage_count + 1U);
            RunFilterFeedbackMatrixBenchmark(configuration, kOrder, stage_count, kBlockSize, bench);
        }
        std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
    }
}
