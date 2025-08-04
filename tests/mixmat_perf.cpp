#include "doctest.h"
#include "nanobench.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "sffdn/sffdn.h"

#include "matrix_multiplication.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("MixMatPerf")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 128;
    constexpr size_t N = 16;

    std::cout << "BLOCK SIZE: " << kBlockSize << std::endl;
    std::cout << "N: " << N << std::endl;

    sfFDN::ScalarFeedbackMatrix mix_mat = sfFDN::ScalarFeedbackMatrix::Householder(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    std::vector<float> output(N * kBlockSize, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());

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

    constexpr size_t block_size = 4;
    constexpr size_t ITER = 94;

    nanobench::Bench bench;
    bench.title("Householder matrix");
    // bench.timeUnit(1ms, "ms");
    bench.warmup(100);

    for (size_t i = 0; i < order.size(); ++i)
    {
        const size_t N = order[i];
        bench.minEpochIterations(4000000 / N);
        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::vector<float> output(N * block_size, 0.f);

        sfFDN::AudioBuffer input_buffer(block_size, N, input.data());
        sfFDN::AudioBuffer output_buffer(block_size, N, output.data());

        sfFDN::ScalarFeedbackMatrix mix_mat = sfFDN::ScalarFeedbackMatrix::Householder(N);
        bench.complexityN(N).run("Householder - Order " + std::to_string(N),
                                 [&] { mix_mat.Process(input_buffer, output_buffer); });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_CASE("FFMPerf_Order")
{
    constexpr size_t N = 8;
    constexpr size_t max_stage = 8;

    constexpr size_t block_size = 512;
    constexpr size_t ITER = 94;

    nanobench::Bench bench;
    bench.title("Filter Feedback Matrix");
    bench.minEpochIterations(10);
    bench.timeUnit(1ms, "ms");
    // bench.relative(true);

    for (size_t i = 1; i < max_stage; ++i)
    {
        size_t K = i;

        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::vector<float> output(N * block_size, 0.f);

        sfFDN::AudioBuffer input_buffer(block_size, N, input.data());
        sfFDN::AudioBuffer output_buffer(block_size, N, output.data());

        auto ffm = CreateFFM(N, i, 1);

        bench.complexityN(i).run("FFM - Stage " + std::to_string(i), [&] {
            ffm->Clear();
            for (size_t i = 0; i < ITER; ++i)
            {
                ffm->Process(input_buffer, output_buffer);
            }
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}
