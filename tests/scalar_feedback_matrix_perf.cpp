#include "nanobench.h"
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
struct MatrixTypeInfo
{
    sfFDN::ScalarMatrixType type;
    std::string_view name;
};

constexpr std::array kMatrixTypes = {
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Identity, "Identity"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Random, "Random"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Householder, "Householder"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::RandomHouseholder, "RandomHouseholder"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Hadamard, "Hadamard"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Circulant, "Circulant"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::Allpass, "Allpass"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::NestedAllpass, "NestedAllpass"},
    MatrixTypeInfo{sfFDN::ScalarMatrixType::VariableDiffusion, "VariableDiffusion"},
};

static_assert(kMatrixTypes.size() == std::to_underlying(sfFDN::ScalarMatrixType::Count));

void RunScalarFeedbackMatrixBenchmark(const MatrixTypeInfo& matrix_type, uint32_t order, uint32_t block_size,
                                      nanobench::Bench& bench)
{
    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());

    sfFDN::test::perf::FillNoise(input);

    sfFDN::ScalarFeedbackMatrix matrix({.matrix_size = order, .type = matrix_type.type});
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);
    const std::string name =
        std::string(matrix_type.name) + " N=" + std::to_string(order) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

std::string_view ExpectedComplexity(sfFDN::ScalarMatrixType type)
{
    if (type == sfFDN::ScalarMatrixType::Householder)
    {
        return "O(n)";
    }
    if (type == sfFDN::ScalarMatrixType::Hadamard)
    {
        return "O(n log n)";
    }
    return "O(n^2)";
}
} // namespace

TEST_CASE("ScalarFeedbackMatrixPerf", "[feedback_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "ScalarFeedbackMatrix perf");
    bool first_benchmark = true;
    for (const MatrixTypeInfo& matrix_type : kMatrixTypes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            for (const uint32_t order : std::array{4u, 8u, 16u, 32u, 64u})
            {
                uint64_t minimum_iterations = 1U;
                if (matrix_type.type == sfFDN::ScalarMatrixType::Householder)
                {
                    minimum_iterations = order == 64U ? 1'200'000U : 7'500'000U / order;
                }
                bench.warmup(first_benchmark ? 500'000U : 100U);
                bench.minEpochIterations(minimum_iterations);
                bench.minEpochTime(matrix_type.type == sfFDN::ScalarMatrixType::Hadamard
                                       ? std::chrono::milliseconds(100)
                                       : std::chrono::milliseconds(10));
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
                RunScalarFeedbackMatrixBenchmark(matrix_type, order, block_size, bench);
                first_benchmark = false;
            }
        }
    }
}

TEST_CASE("ScalarFeedbackMatrixPerf_Aliased", "[feedback_matrix]")
{
    constexpr uint32_t kBlockSize = 128U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "ScalarFeedbackMatrix aliased perf");

    for (const MatrixTypeInfo& matrix_type : kMatrixTypes)
    {
        for (const uint32_t order : sfFDN::test::perf::kChannelCounts)
        {
            const bool needs_iteration_floor =
                matrix_type.type == sfFDN::ScalarMatrixType::Allpass && order == 32U;
            bench.minEpochIterations(needs_iteration_floor ? 25'000U : 1U);
            std::vector<float> inout(static_cast<size_t>(order) * kBlockSize);
            sfFDN::test::perf::FillNoise(inout);
            sfFDN::ScalarFeedbackMatrix matrix({.matrix_size = order, .type = matrix_type.type});
            sfFDN::AudioBuffer buffer(kBlockSize, order, inout);
            sfFDN::test::perf::SetChannelSampleBatch(bench, kBlockSize, order);

            bench.run(std::string(matrix_type.name) + " N=" + std::to_string(order), [&] {
                matrix.Process(buffer, buffer);
                nanobench::doNotOptimizeAway(inout);
            });
        }
    }
}

TEST_CASE("ScalarFeedbackMatrixPerf_BigO", "[feedback_matrix][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128;
    for (const MatrixTypeInfo& matrix_type : kMatrixTypes)
    {
        nanobench::Bench bench;
        sfFDN::test::perf::ConfigureComplexityBench(
            bench,
            std::string("ScalarFeedbackMatrix ") + std::string(matrix_type.name) + " B=" + std::to_string(kBlockSize));

        for (const uint32_t order : std::array{4u, 8u, 16u, 32u, 64u})
        {
            bench.complexityN(order);
            RunScalarFeedbackMatrixBenchmark(matrix_type, order, kBlockSize, bench);
        }

        const auto fits = bench.complexityBigO();
        std::cout << sfFDN::test::perf::FormatComplexityFits(fits) << '\n';
        if (sfFDN::test::perf::ComplexityEnforcementEnabled())
        {
            const std::string_view expected_complexity = ExpectedComplexity(matrix_type.type);
            CAPTURE(matrix_type.name, kBlockSize, expected_complexity);
            REQUIRE_FALSE(fits.empty());
            INFO("Complexity fits:\n" << fits);
            CHECK(fits.front().name() == expected_complexity);
        }
    }
}