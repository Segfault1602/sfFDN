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

    std::cout << "BLOCK SIZE: " << kBlockSize << std::endl;
    std::cout << "N: " << N << std::endl;

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
    bench.minEpochIterations(100000);
    bench.timeUnit(1us, "us");

    bench.run("Householder", [&] { mix_mat.Process(input_buffer, input_buffer); });

    // auto ffm = CreateFFM(N, 4, 3);
    // bench.minEpochIterations(209);
    // bench.run("FFM", [&] { ffm->Tick(input, output); });

    std::filesystem::path output_dir = std::filesystem::current_path() / "perf";
    if (!std::filesystem::exists(output_dir))
    {
        std::filesystem::create_directory(output_dir);
    }
    std::filesystem::path filepath = output_dir / std::format("matrix_B{}.json", kBlockSize);
    std::ofstream render_out(filepath);
    bench.render(ankerl::nanobench::templates::json(), render_out);
}

TEST_CASE("Matrix_Order")
{
    constexpr std::array order = {4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32, 64, 128};

    constexpr uint32_t block_size = 4;
    constexpr uint32_t ITER = 94;

    nanobench::Bench bench;
    bench.title("Householder matrix");
    // bench.timeUnit(1ms, "ms");
    bench.warmup(100);

    sfFDN::RNG rng;
    for (auto i = 0; i < order.size(); ++i)
    {
        const uint32_t N = order[i];
        bench.minEpochIterations(4000000 / N);
        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = rng.NextFloat();
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

    constexpr uint32_t block_size = 512;
    constexpr uint32_t ITER = 94;

    nanobench::Bench bench;
    bench.title("Filter Feedback Matrix");
    bench.minEpochIterations(10);
    bench.timeUnit(1ms, "ms");
    // bench.relative(true);

    sfFDN::RNG rng;
    for (auto i = 1; i < max_stage; ++i)
    {
        uint32_t K = i;

        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = rng.NextFloat();
        }
        std::vector<float> output(N * block_size, 0.f);

        sfFDN::AudioBuffer input_buffer(block_size, N, input);
        sfFDN::AudioBuffer output_buffer(block_size, N, output);

        auto ffm = CreateFFM(N, i, 1);

        bench.complexityN(i).run("FFM - Stage " + std::to_string(i), [&] {
            ffm->Clear();
            for (auto i = 0; i < ITER; ++i)
            {
                ffm->Process(input_buffer, output_buffer);
            }
        });
    }

    std::cout << bench.complexityBigO() << "\n";
}
