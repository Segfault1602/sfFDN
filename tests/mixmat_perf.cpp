#include "doctest.h"
#include "nanobench.h"

#include <iostream>
#include <random>

#include "mixing_matrix.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("MixMatPerf")
{
    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = ((SR / block_size) + 1); // 1 second at 48kHz
    constexpr size_t N = 16;

    std::cout << "ITER: " << ITER << std::endl;
    std::cout << "BLOCK SIZE: " << block_size << std::endl;
    std::cout << "N: " << N << std::endl;

    fdn::MixMat mix_mat = fdn::MixMat::Householder(N);

    std::vector<float> input(N * block_size, 0.f);
    std::vector<float> output(N * block_size, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("Householder matrix");
    // bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");

    bench.run("Householder", [&] {
        for (size_t i = 0; i < ITER; ++i)
        {
            mix_mat.Tick(input, output);
        }
    });
}

TEST_CASE("Matrix_Order")
{
    constexpr std::array<size_t, 9> order = {4, 6, 8, 10, 12, 14, 16, 24, 32};

    constexpr size_t block_size = 512;
    constexpr size_t ITER = 94;

    nanobench::Bench bench;
    bench.title("Householder matrix");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    // bench.warmup(100);

    for (size_t i = 0; i < order.size(); ++i)
    {
        const size_t N = order[i];
        // fill input with random values
        std::vector<float> input(N * block_size, 0.f);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::vector<float> output(N * block_size, 0.f);

        bench.complexityN(N).run("Householder - Order " + std::to_string(N), [&] {
            fdn::MixMat mix_mat = fdn::MixMat::Householder(N);
            for (size_t i = 0; i < ITER; ++i)
            {
                mix_mat.Tick(input, output);
            }
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_CASE("FFMPerf_Order")
{
    constexpr size_t N = 6;
    constexpr size_t max_stage = 16;

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

        auto ffm = CreateFFM(N, i, 1);

        bench.complexityN(i).run("FFM - Stage " + std::to_string(i), [&] {
            ffm->Clear();
            for (size_t i = 0; i < ITER; ++i)
            {
                ffm->Tick(input, output);
            }
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}