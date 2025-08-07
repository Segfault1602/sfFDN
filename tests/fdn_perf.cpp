#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>

#include "array_math.h"
#include "filter_coeffs.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FDNPerf")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;

    auto fdn = CreateFDN(SR, kBlockSize, N);

    std::vector<float> input(kBlockSize * N, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    bench.run("FDN", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });

    // Benchmark the individual components
    auto input_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Multiplexed, std::vector<float>(N, 1.f));
    bench.run("Input Gains", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        input_gains->Process(input_buffer, output_buffer);
    });

    sfFDN::DelayBank delay_bank(GetDefaultDelays(N), kBlockSize);
    bench.run("Delay Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        delay_bank.GetNextOutputs(output_buffer);
        delay_bank.AddNextInputs(input_buffer);
    });

    auto filter_bank = GetFilterBank(N, 11);
    bench.run("Filter Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        filter_bank->Process(input_buffer, output_buffer);
    });

    auto fir_filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0; i < N; i++)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_1153.wav");
        auto PartitionedConvolver = std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, fir);

        fir_filter_bank->AddFilter(std::move(PartitionedConvolver));
    }
    bench.run("FIR Filter Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        fir_filter_bank->Process(input_buffer, output_buffer);
    });

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(sfFDN::ScalarFeedbackMatrix::Householder(N));
    bench.run("Mixing Matrix", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        mix_mat->Process(input_buffer, output_buffer);
    });

    auto output_gains = GetDefaultOutputGains(N);
    bench.run("Output Gains", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        output_gains->Process(input_buffer, output_buffer);
    });

    auto tc_filter = GetDefaultTCFilter();
    bench.run("TC Filter", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        tc_filter->Process(input_buffer, output_buffer);
    });

    bench.run("Direct Gain", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        sfFDN::ArrayMath::ScaleAccumulate(input_buffer.GetChannelSpan(0), 1.f, output_buffer.GetChannelSpan(0));
    });
}

TEST_CASE("FDNPerf_FIR")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;

    auto fdn = CreateFDN(SR, kBlockSize, N);

    // Replace filterbank with FIR filters
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0; i < N; i++)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_1153.wav");
        auto convolver = std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, fir);

        filter_bank->AddFilter(std::move(convolver));
    }

    fdn->SetFilterBank(std::move(filter_bank));

    std::vector<float> input(kBlockSize * N, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("FDN Perf - FIR");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    bench.run("FDN", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });
}

TEST_CASE("FDNPerf_FFM")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    constexpr uint32_t K = 4;
    std::array<uint32_t, N*(K - 1)> ffm_delays = {
        2, 3, 8, 10, 14, 16, 0, 18, 36, 54, 72, 90, 0, 108, 216, 324, 432, 540,
    };

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(N);

    std::vector<sfFDN::ScalarFeedbackMatrix> mixing_matrices(K);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    for (uint32_t i = 0; i < K; ++i)
    {
        float u_n[N] = {0.f};
        for (uint32_t j = 0; j < N; ++j)
        {
            u_n[j] = dis(gen);
        }

        mixing_matrices[i] = sfFDN::ScalarFeedbackMatrix::Householder(u_n);
    }
    ffm->ConstructMatrix(ffm_delays, mixing_matrices);

    auto fdn = CreateFDN(SR, kBlockSize, N);
    fdn->SetFeedbackMatrix(std::move(ffm));

    nanobench::Bench bench;
    bench.title("FDN Perf");
    bench.timeUnit(1us, "us");
    bench.run("FDN_FFM", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });
}

TEST_CASE("FDNPerf_Order")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;

    constexpr uint32_t order[] = {4, 8, 16, 32, 64};

    nanobench::Bench bench;
    bench.title("FDN Perf - Order");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    for (uint32_t i = 0; i < sizeof(order) / sizeof(order[0]); ++i)
    {
        uint32_t N = order[i];
        auto fdn = CreateFDN(SR, kBlockSize, N);

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(N).run("FDN Order " + std::to_string(N), [&] {
            sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}

TEST_CASE("FDNPerf_BlockSize")
{
    constexpr uint32_t SR = 48000;
    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr uint32_t kInputSize = 1 << 12;
    constexpr uint32_t kOrder = 16;

    nanobench::Bench bench;
    bench.title("FDN Perf - Order");
    bench.relative(true);
    bench.warmup(10);
    bench.batch(kInputSize);
    bench.minEpochIterations(100);

    std::vector<float> input(kInputSize, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(kInputSize, 0.f);

    for (uint32_t i = 0; i < kBlockSizes.size(); ++i)
    {
        const uint32_t block_size = kBlockSizes[i];
        auto fdn = CreateFDN(SR, block_size, kOrder);

        bench.run("FDN Block Size " + std::to_string(block_size), [&] {
            uint32_t block_count = kInputSize / block_size;
            for (auto i = 0; i < block_count; ++i)
            {
                sfFDN::AudioBuffer input_buffer(block_size, 1, input.data() + i * block_size);
                sfFDN::AudioBuffer output_buffer(block_size, 1, output.data() + i * block_size);
                fdn->Process(input_buffer, output_buffer);
            }
        });
    }
}

TEST_CASE("FDNPerf_OrderFFM")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 512;

    constexpr std::array<uint32_t, 5> num_stages = {2, 3, 4, 5, 6};

    nanobench::Bench bench;
    bench.title("FDN Perf - FFM");
    // bench.minEpochIterations(100);
    bench.batch(kBlockSize);

    for (auto i = 0; i < num_stages.size(); ++i)
    {
        constexpr uint32_t N = 8;
        auto fdn = CreateFDN(SR, kBlockSize, N);

        auto ffm = CreateFFM(N, num_stages[i], 1);
        fdn->SetFeedbackMatrix(std::move(ffm));

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        std::default_random_engine generator;
        std::normal_distribution<double> dist(0, 0.1);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = dist(generator);
        }

        bench.complexityN(num_stages[i]).run("FFM num stages: " + std::to_string(num_stages[i]), [&] {
            sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << std::endl;
}
