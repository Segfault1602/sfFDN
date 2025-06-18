#include "doctest.h"
#include "nanobench.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>

#include "array_math.h"
#include "fdn.h"
#include "feedback_matrix.h"
#include "filter_coeffs.h"
#include "filter_design.h"
#include "filter_feedback_matrix.h"
#include "nupols.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_SUITE_BEGIN("FDN");

TEST_CASE("FDNPerf")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 128;
    constexpr size_t N = 16;

    auto fdn = CreateFDN(SR, kBlockSize, N);

    std::vector<float> input(kBlockSize * N, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    bench.run("FDN", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });

    // Benchmark the individual components
    auto input_gains =
        std::make_unique<fdn::ParallelGains>(fdn::ParallelGainsMode::Multiplexed, std::vector<float>(N, 1.f));
    bench.run("Input Gains", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, N, output);
        input_gains->Process(input_buffer, output_buffer);
    });

    fdn::DelayBank delay_bank(GetDefaultDelays(N), kBlockSize);
    bench.run("Delay Bank", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, N, input);
        fdn::AudioBuffer output_buffer(kBlockSize, N, output);
        delay_bank.GetNextOutputs(output_buffer);
        delay_bank.AddNextInputs(input_buffer);
    });

    auto filter_bank = GetFilterBank(N, 11);
    bench.run("Filter Bank", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, N, input);
        fdn::AudioBuffer output_buffer(kBlockSize, N, output);
        filter_bank->Process(input_buffer, output_buffer);
    });

    auto fir_filter_bank = std::make_unique<fdn::FilterBank>();
    for (size_t i = 0; i < N; i++)
    {
        auto fir = ReadWavFile("./tests/att_fir_1153.wav");
        auto nupols = std::make_unique<fdn::NUPOLS>(kBlockSize, fir, fdn::PartitionStrategy::kGardner);

        fir_filter_bank->AddFilter(std::move(nupols));
    }
    bench.run("FIR Filter Bank", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, N, input);
        fdn::AudioBuffer output_buffer(kBlockSize, N, output);
        fir_filter_bank->Process(input_buffer, output_buffer);
    });

    auto mix_mat = std::make_unique<fdn::ScalarFeedbackMatrix>(fdn::ScalarFeedbackMatrix::Householder(N));
    bench.run("Mixing Matrix", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, N, input);
        fdn::AudioBuffer output_buffer(kBlockSize, N, output);
        mix_mat->Process(input_buffer, output_buffer);
    });

    auto output_gains = GetDefaultOutputGains(N);
    bench.run("Output Gains", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, N, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        output_gains->Process(input_buffer, output_buffer);
    });

    auto tc_filter = GetDefaultTCFilter();
    bench.run("TC Filter", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        tc_filter->Process(input_buffer, output_buffer);
    });

    bench.run("Direct Gain", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn::ArrayMath::ScaleAccumulate(input_buffer.GetChannelSpan(0), 1.f, output_buffer.GetChannelSpan(0));
    });
}

TEST_CASE("FDNPerf_FIR")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 128;
    constexpr size_t N = 16;

    auto fdn = CreateFDN(SR, kBlockSize, N);

    // Replace filterbank with FIR filters
    auto filter_bank = std::make_unique<fdn::FilterBank>();
    for (size_t i = 0; i < N; i++)
    {
        auto fir = ReadWavFile("./tests/att_fir_1153.wav");
        auto nupols = std::make_unique<fdn::NUPOLS>(kBlockSize, fir, fdn::PartitionStrategy::kGardner);

        filter_bank->AddFilter(std::move(nupols));
    }

    fdn->SetFilterBank(std::move(filter_bank));

    std::vector<float> input(kBlockSize * N, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf - FIR");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    bench.run("FDN", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });
}

TEST_CASE("FDNPerf_FFM")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 128;
    constexpr size_t N = 16;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    constexpr size_t K = 4;
    std::array<size_t, N*(K - 1)> ffm_delays = {
        2, 3, 8, 10, 14, 16, 0, 18, 36, 54, 72, 90, 0, 108, 216, 324, 432, 540,
    };

    auto ffm = std::make_unique<fdn::FilterFeedbackMatrix>(N);

    std::vector<fdn::ScalarFeedbackMatrix> mixing_matrices(K);
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

        mixing_matrices[i] = fdn::ScalarFeedbackMatrix::Householder(u_n);
    }
    ffm->ConstructMatrix(ffm_delays, mixing_matrices);

    auto fdn = CreateFDN(SR, kBlockSize, N);
    fdn->SetFeedbackMatrix(std::move(ffm));

    nanobench::Bench bench;
    bench.title("FDN Perf");
    bench.timeUnit(1us, "us");
    bench.run("FDN_FFM", [&] {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });
}

TEST_CASE("FDNPerf_Order")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 128;

    constexpr size_t order[] = {4, 8, 16, 32, 64};

    nanobench::Bench bench;
    bench.title("FDN Perf - Order");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);
    // bench.batch(kBlockSize);

    for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); ++i)
    {
        size_t N = order[i];
        auto fdn = CreateFDN(SR, kBlockSize, N);

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(N).run("FDN Order " + std::to_string(N), [&] {
            fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
            fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_CASE("FDNPerf_OrderFFM")
{
    constexpr size_t SR = 48000;
    constexpr size_t kBlockSize = 512;

    constexpr std::array<size_t, 5> num_stages = {2, 3, 4, 5, 6};

    nanobench::Bench bench;
    bench.title("FDN Perf - FFM");
    // bench.minEpochIterations(100);
    bench.batch(kBlockSize);

    for (size_t i = 0; i < num_stages.size(); ++i)
    {
        constexpr size_t N = 8;
        auto fdn = CreateFDN(SR, kBlockSize, N);

        auto ffm = CreateFFM(N, num_stages[i], 1);
        fdn->SetFeedbackMatrix(std::move(ffm));

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (size_t i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(num_stages[i]).run("FFM num stages: " + std::to_string(num_stages[i]), [&] {
            fdn::AudioBuffer input_buffer(kBlockSize, 1, input);
            fdn::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_SUITE_END();