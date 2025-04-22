#include "doctest.h"
#include "nanobench.h"

#include <iostream>

#include "mixing_matrix.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("Perf" * doctest::skip(true))
{
    constexpr size_t N = 16;
    fdn::MixMat mix_mat = fdn::MixMat::Householder(N);

    constexpr size_t size = 48000;
    std::vector<float> input(N * size);

    // fill input with random values
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    nanobench::Bench bench;
    bench.title("Householder matrix");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    bench.relative(true);

    std::vector<float> output(N * size);
    bench.run("Householder - All", [&] {
        mix_mat.Tick(input, output);
        nanobench::doNotOptimizeAway(output);
    });

    std::vector<float> output_sample(N * size);
    constexpr size_t block_size[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, size};

    for (size_t bs : block_size)
    {
        bench.run("Householder - Block size " + std::to_string(bs), [&] {
            for (size_t i = 0; i < input.size(); i += N * bs)
            {
                std::span<float> input_span(input.data() + i, N * bs);
                std::span<float> output_span(output_sample.data() + i, N * bs);
                mix_mat.Tick(input_span, output_span);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(output_sample[i]).epsilon(0.0001));
    }
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