#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "matrix_multiplication.h"
#include "rng.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("MixMatPerf")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;

    sfFDN::ScalarFeedbackMatrix mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Householder);

    std::vector<float> input(N * kBlockSize, 0.f);
    std::vector<float> output(N * kBlockSize, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);

    nanobench::Bench bench;
    bench.title("Householder matrix");
    // bench.batch(kBlockSize);
    bench.minEpochIterations(10000);
    bench.timeUnit(1us, "us");

    bench.run("Householder", [&] { mix_mat.Process(input_buffer, input_buffer); });
}

TEST_CASE("Matrix_Order")
{
    constexpr std::array order = {4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32, 64, 128};

    constexpr uint32_t block_size = 128;

    nanobench::Bench bench;
    bench.title("Householder matrix - Complexity");

    sfFDN::RNG rng;
    for (unsigned int N : order)
    {
        bench.minEpochIterations(200000 / (N * N));
        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (float& i : input)
        {
            i = rng.NextFloat();
        }
        std::vector<float> output(N * block_size, 0.f);

        sfFDN::AudioBuffer input_buffer(block_size, N, input);
        sfFDN::AudioBuffer output_buffer(block_size, N, output);

        sfFDN::ScalarFeedbackMatrix mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Householder);
        bench.complexityN(N).run("Householder - Order " + std::to_string(N),
                                 [&] { mix_mat.Process(input_buffer, output_buffer); });
    }

    std::cout << bench.complexityBigO() << "\n";
}

TEST_CASE("FFMPerf_Order")
{
    constexpr uint32_t N = 8;
    constexpr uint32_t max_stage = 8;

    constexpr uint32_t block_size = 128;

    nanobench::Bench bench;
    bench.title("Filter Feedback Matrix");
    bench.minEpochIterations(10);
    bench.timeUnit(1ms, "ms");
    // bench.relative(true);

    // fill input with random values
    sfFDN::RNG rng;
    std::vector<float> input(N * block_size, 0.f);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = rng.NextFloat();
    }
    std::vector<float> output(N * block_size, 0.f);

    sfFDN::AudioBuffer input_buffer(block_size, N, input);
    sfFDN::AudioBuffer output_buffer(block_size, N, output);

    for (auto i = 1; i < max_stage; ++i)
    {
        auto ffm = CreateFFM(N, i, 1);
        bench.complexityN(i).run("FFM - Stage " + std::to_string(i),
                                 [&] { ffm->Process(input_buffer, output_buffer); });
    }

    std::cout << bench.complexityBigO() << "\n";
}
