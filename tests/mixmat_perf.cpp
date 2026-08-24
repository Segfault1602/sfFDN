#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string_view>

#include "matrix_multiplication.h"
#include "rng.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
void BenchmarkStructuredMatrix(const sfFDN::ScalarMatrixType type, const std::string_view name, const uint32_t order,
                               const uint32_t block_size, nanobench::Bench& bench)
{
    std::vector<float> input(order * block_size);
    std::vector<float> structured_output(input.size());
    std::vector<float> dense_output(input.size());

    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    const auto matrix_data = sfFDN::GenerateMatrix(order, type);
    sfFDN::ScalarFeedbackMatrix structured({.matrix_size = order, .type = type});
    sfFDN::ScalarFeedbackMatrix dense({.matrix_size = order, .type = type, .custom_matrix = matrix_data});
    sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer structured_buffer(block_size, order, structured_output);
    sfFDN::AudioBuffer dense_buffer(block_size, order, dense_output);
    const std::string suffix = " o" + std::to_string(order) + " b" + std::to_string(block_size);

    bench.run(std::string(name) + " structured" + suffix, [&] {
        structured.Process(input_buffer, structured_buffer);
        nanobench::doNotOptimizeAway(structured_output);
    });
    bench.run(std::string(name) + " dense" + suffix, [&] {
        dense.Process(input_buffer, dense_buffer);
        nanobench::doNotOptimizeAway(dense_output);
    });
}
} // namespace

TEST_CASE("MixMatPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kMatSize = 16;
    constexpr uint32_t kInputSize = kMatSize * kBlockSize;

    sfFDN::ScalarFeedbackMatrix mix_mat =
        sfFDN::ScalarFeedbackMatrix({.matrix_size = kMatSize, .type = sfFDN::ScalarMatrixType::Householder});

    std::array<float, kInputSize> input{};
    std::array<float, kInputSize> output{};
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    nanobench::Bench bench;
    bench.title("Householder matrix");
    // bench.batch(kBlockSize);
    bench.minEpochIterations(100000);
    bench.timeUnit(1us, "us");

    bench.run("Householder", [&] {
        mix_mat.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::ScalarFeedbackMatrix random_mat =
        sfFDN::ScalarFeedbackMatrix({.matrix_size = kMatSize, .type = sfFDN::ScalarMatrixType::Random});

    bench.run("Random", [&] {
        random_mat.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

    auto hadamard = sfFDN::ScalarFeedbackMatrix({.matrix_size = kMatSize, .type = sfFDN::ScalarMatrixType::Hadamard});
    bench.run("Hadamard", [&] {
        hadamard.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("StructuredMixMatPerf")
{
    nanobench::Bench bench;
    bench.title("Structured feedback matrix");
    bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochIterations(20000);

    for (const uint32_t order : {4u, 8u, 16u, 32u})
    {
        for (const uint32_t block_size : {64u, 128u, 256u})
        {
            BenchmarkStructuredMatrix(sfFDN::ScalarMatrixType::Hadamard, "Hadamard", order, block_size, bench);
            BenchmarkStructuredMatrix(sfFDN::ScalarMatrixType::Householder, "Householder", order, block_size, bench);
        }
    }
}

TEST_CASE("Matrix_Order")
{
    constexpr std::array<uint32_t, 14> kMatrixSizes = {4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32, 64, 128};

    constexpr uint32_t kBlockSize = 128;

    nanobench::Bench bench;
    bench.title("Householder matrix - Complexity");
    bench.timeUnit(1us, "us");

    sfFDN::RNG rng;
    for (auto mat_size : kMatrixSizes)
    {
        bench.minEpochIterations(200000 / (mat_size * mat_size));
        // fill input with random values
        std::vector<float> input(mat_size * kBlockSize, 0.f);
        for (float& i : input)
        {
            i = rng();
        }
        std::vector<float> output(mat_size * kBlockSize, 0.f);

        sfFDN::AudioBuffer input_buffer(kBlockSize, mat_size, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, mat_size, output);

        sfFDN::ScalarFeedbackMatrix mix_mat =
            sfFDN::ScalarFeedbackMatrix({mat_size, sfFDN::ScalarMatrixType::Householder});
        bench.run("Householder - Order " + std::to_string(mat_size),
                  [&] { mix_mat.Process(input_buffer, output_buffer); });
    }

    // std::cout << bench.complexityBigO() << "\n";
}

TEST_CASE("FFMPerf_Order")
{
    constexpr uint32_t kMatSize = 8;
    constexpr uint32_t kMaxStageCount = 8;

    constexpr uint32_t kBlockSize = 128;

    nanobench::Bench bench;
    bench.title("Filter Feedback Matrix");
    bench.timeUnit(1us, "us");
    // bench.relative(true);

    // fill input with random values
    sfFDN::RNG rng;
    std::vector<float> input(kMatSize * kBlockSize, 0.f);
    for (auto& i : input)
    {
        i = rng();
    }
    std::vector<float> output(kMatSize * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    for (auto i = 1; i < kMaxStageCount; ++i)
    {
        bench.minEpochIterations(10000 / i);
        auto ffm = CreateFFM(kMatSize, i, 1);
        bench.complexityN(i).run("FFM - Stage " + std::to_string(i),
                                 [&] { ffm->Process(input_buffer, output_buffer); });
    }

    std::cout << bench.complexityBigO() << "\n";
}

TEST_CASE("Delay_Matrix")
{
    constexpr std::array<uint32_t, 12> kMatrixSizes = {4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32};

    constexpr uint32_t kBlockSize = 128;

    nanobench::Bench bench;
    bench.title("Filter Feedback Matrix");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);
    // bench.relative(true);

    for (auto mat_size : kMatrixSizes)
    {
        // fill input with random values
        sfFDN::RNG rng;
        std::vector<float> input(mat_size * kBlockSize, 0.f);
        for (auto& i : input)
        {
            i = rng();
        }
        std::vector<float> output(mat_size * kBlockSize, 0.f);

        sfFDN::AudioBuffer input_buffer(kBlockSize, mat_size, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, mat_size, output);

        std::vector<uint32_t> delays(mat_size * mat_size, 0);
        for (auto& d : delays)
        {
            d = std::abs(rng()) * 20; // random delay between 0 and 1000 samples
        }

        sfFDN::ScalarFeedbackMatrix mixing_matrix =
            sfFDN::ScalarFeedbackMatrix({mat_size, sfFDN::ScalarMatrixType::Hadamard});
        sfFDN::DelayMatrix delay_matrix(mat_size, delays, mixing_matrix);

        bench.run("Delay Matrix - Order " + std::to_string(mat_size),
                  [&] { delay_matrix.Process(input_buffer, output_buffer); });
    }
}