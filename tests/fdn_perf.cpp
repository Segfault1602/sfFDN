#include "doctest.h"
#include "nanobench.h"

#include <iostream>
#include <random>
#include <span>

#include "fdn.h"
#include "filter_coeffs.h"
#include "filter_design.h"
#include "filter_feedback_matrix.h"
#include "mixing_matrix.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FDNPerf")
{
    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = ((SR / block_size) + 1) * block_size; // 1 second at 48kHz
    constexpr size_t N = 16;

    std::cout << "ITER: " << ITER << std::endl;
    std::cout << "BLOCK SIZE: " << block_size << std::endl;
    std::cout << "N: " << N << std::endl;

    auto fdn = CreateFDN(SR, block_size, N);

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf");
    // bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    bench.relative(true);
    // bench.batch(ITER);
    // bench.unit("block");

    bench.run("FDN", [&] {
        fdn->Clear();
        for (size_t i = 0; i < input.size(); i += block_size)
        {
            std::span<float> input_span{input.data() + i, block_size};
            std::span<float> output_span{output.data() + i, block_size};
            fdn->Tick(input_span, output_span);
        }
    });

    constexpr size_t K = 4;
    std::array<size_t, N*(K - 1)> ffm_delays = {
        2, 3, 8, 10, 14, 16, 0, 18, 36, 54, 72, 90, 0, 108, 216, 324, 432, 540,
    };

    auto ffm = std::make_unique<fdn::FilterFeedbackMatrix>(N, K);
    ffm->SetDelays(ffm_delays);

    std::vector<fdn::MixMat> mixing_matrices(K);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    for (size_t i = 0; i < K; ++i)
    {
        float u_n[N] = {0.f};
        for (size_t j = 0; j < N; ++j)
        {
            u_n[j] = dis(gen);
        }

        mixing_matrices[i] = fdn::MixMat::Householder(u_n);
    }
    ffm->SetMatrices(mixing_matrices);

    fdn = CreateFDN(SR, block_size, N);
    fdn->SetFeedbackMatrix(std::move(ffm));

    bench.run("FDN_FFM", [&] {
        fdn->Clear();
        for (size_t i = 0; i < input.size(); i += block_size)
        {
            std::span<float> input_span{input.data() + i, block_size};
            std::span<float> output_span{output.data() + i, block_size};
            fdn->Tick(input_span, output_span);
        }
    });
}

TEST_CASE("FDNPerf_Order")
{
    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = 94 * block_size; // 1 second at 48kHz

    constexpr size_t order[] = {4, 6, 8, 10, 12, 14, 16, 24, 32};

    nanobench::Bench bench;
    bench.title("FDN Perf");
    bench.timeUnit(1ms, "ms");
    bench.performanceCounters(true);

    for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); ++i)
    {
        size_t N = order[i];
        auto fdn = CreateFDN(SR, block_size, N);

        std::vector<float> input(ITER, 0.f);
        std::vector<float> output(ITER, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(N).run("FDN Order " + std::to_string(N), [&] {
            fdn->Clear();
            for (size_t i = 0; i < input.size(); i += block_size)
            {
                std::span<float> input_span{input.data() + i, block_size};
                std::span<float> output_span{output.data() + i, block_size};
                fdn->Tick(input_span, output_span);
            }
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_CASE("FDNPerf_OrderFFM")
{
    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = 94 * block_size; // 1 second at 48kHz

    constexpr std::array<size_t, 15> num_stages = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    nanobench::Bench bench;
    bench.title("FDN Perf - FFM");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    // bench.relative(true);
    // bench.batch(ITER);
    // bench.unit("block");

    for (size_t i = 0; i < num_stages.size(); ++i)
    {
        constexpr size_t N = 8;
        auto fdn = CreateFDN(SR, block_size, N);

        auto ffm = CreateFFM(N, num_stages[i], 1);
        fdn->SetFeedbackMatrix(std::move(ffm));

        std::vector<float> input(ITER, 0.f);
        std::vector<float> output(ITER, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(num_stages[i]).run("FFM num stages: " + std::to_string(num_stages[i]), [&] {
            fdn->Clear();
            for (size_t i = 0; i < input.size(); i += block_size)
            {
                std::span<float> input_span{input.data() + i, block_size};
                std::span<float> output_span{output.data() + i, block_size};
                fdn->Tick(input_span, output_span);
            }
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}