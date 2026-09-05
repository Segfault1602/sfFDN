#include "nanobench.h"
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
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
} // namespace

TEST_CASE("ScalarFeedbackMatrixPerf", "[feedback_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "ScalarFeedbackMatrix perf");
    for (const MatrixTypeInfo& matrix_type : kMatrixTypes)
    {
        for (const uint32_t block_size : std::array{32u, 64u, 128u, 256u})
        {
            for (const uint32_t order : std::array{4u, 8u, 16u, 32u, 64u})
            {
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
                RunScalarFeedbackMatrixBenchmark(matrix_type, order, block_size, bench);
            }
        }
    }
}

TEST_CASE("ScalarFeedbackMatrixPerf_BigO", "[feedback_matrix]")
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

        std::string_view expected_complexity = "O(n^2)";
        if (matrix_type.type == sfFDN::ScalarMatrixType::Householder)
        {
            expected_complexity = "O(n)";
        }
        else if (matrix_type.type == sfFDN::ScalarMatrixType::Hadamard)
        {
            expected_complexity = "O(n log n)";
        }

        const auto fits = bench.complexityBigO();
        CAPTURE(matrix_type.name, kBlockSize, expected_complexity);
        REQUIRE_FALSE(fits.empty());
        INFO("Complexity fits:\n" << fits);
        CHECK(fits.front().name() == expected_complexity);
    }
}