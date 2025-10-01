#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>

#include "sffdn/sffdn.h"

#include "array_math.h"
#include "filter_coeffs.h"
#include "rng.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FDNPerf")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kFDNOrder = 16;

    auto fdn = CreateFDN(kBlockSize, kFDNOrder);

    std::vector<float> input(kBlockSize * kFDNOrder, 0.f);
    std::vector<float> output(kBlockSize * kFDNOrder, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    nanobench::Bench bench;
    bench.title("FDN Perf");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    bench.run("FDN", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });

    // Benchmark the individual components
    auto input_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split, std::vector<float>(kFDNOrder, 1.f));
    bench.minEpochIterations(50000);
    bench.run("Input Gains", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kFDNOrder, output);
        input_gains->Process(input_buffer, output_buffer);
    });

    sfFDN::DelayBank delay_bank(GetDefaultDelays(kFDNOrder), kBlockSize);
    bench.run("Delay Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kFDNOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kFDNOrder, output);
        delay_bank.GetNextOutputs(output_buffer);
        delay_bank.AddNextInputs(input_buffer);
    });

    bench.minEpochIterations(1000);
    auto filter_bank = GetFilterBank(kFDNOrder, 11);
    bench.run("Filter Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kFDNOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kFDNOrder, output);
        filter_bank->Process(input_buffer, output_buffer);
    });

    auto fir_filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0u; i < kFDNOrder; i++)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_1153.wav");
        fir_filter_bank->AddFilter(std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, fir));
    }
    bench.minEpochIterations(100);
    bench.run("FIR Filter Bank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kFDNOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kFDNOrder, output);
        fir_filter_bank->Process(input_buffer, output_buffer);
    });

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(
        sfFDN::ScalarFeedbackMatrix(kFDNOrder, sfFDN::ScalarMatrixType::Householder));
    bench.run("Mixing Matrix", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kFDNOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kFDNOrder, output);
        mix_mat->Process(input_buffer, output_buffer);
    });

    auto output_gains = GetDefaultOutputGains(kFDNOrder);
    bench.run("Output Gains", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kFDNOrder, input);
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
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kFDNOrder = 16;

    auto fdn = CreateFDN(kBlockSize, kFDNOrder);

    // Replace filterbank with FIR filters
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0u; i < kFDNOrder; i++)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_1153.wav");
        auto convolver = std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, fir);

        filter_bank->AddFilter(std::move(convolver));
    }

    fdn->SetFilterBank(std::move(filter_bank));

    std::vector<float> input(kBlockSize * kFDNOrder, 0.f);
    std::vector<float> output(kBlockSize * kFDNOrder, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    nanobench::Bench bench;
    bench.title("FDN Perf - FIR");
    // bench.batch(kBlockSize);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    bench.run("FDN", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        fdn->Process(input_buffer, output_buffer);
    });
}

TEST_CASE("FDNPerf_FFM")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kFDNOrder = 16;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    constexpr uint32_t kStageCount = 4;
    std::array<uint32_t, kFDNOrder*(kStageCount - 1)> ffm_delays = {
        2, 3, 8, 10, 14, 16, 0, 18, 36, 54, 72, 90, 0, 108, 216, 324, 432, 540,
    };

    sfFDN::CascadedFeedbackMatrixInfo ffm_info =
        sfFDN::ConstructCascadedFeedbackMatrix(kFDNOrder, kStageCount, 1, sfFDN::ScalarMatrixType::Hadamard);

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(ffm_info);

    auto fdn = CreateFDN(kBlockSize, kFDNOrder);
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
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;

    constexpr std::array<uint32_t, 5> kFDNOrder = {4, 8, 16, 32, 64};

    nanobench::Bench bench;
    bench.title("FDN Perf - Order");
    bench.timeUnit(1us, "us");

    for (auto fdn_order : kFDNOrder)
    {
        auto fdn = CreateFDN(kBlockSize, fdn_order);

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        sfFDN::RNG generator;
        for (auto& i : input)
        {
            i = generator();
        }

        bench.minEpochIterations(40000 / fdn_order);
        bench.complexityN(fdn_order).run("FDN Order " + std::to_string(fdn_order), [&] {
            sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << "\n";
}

TEST_CASE("FDNPerf_BlockSize")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr uint32_t kInputSize = 2048;
    constexpr uint32_t kOrder = 16;

    nanobench::Bench bench;
    bench.title("FDN Perf - Block Size");
    bench.relative(true);
    bench.warmup(10);
    // bench.batch(kInputSize);
    bench.minEpochIterations(100);

    std::vector<float> input(kInputSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    std::vector<float> output(kInputSize, 0.f);

    for (unsigned int block_size : kBlockSizes)
    {
        auto fdn = CreateFDN(block_size, kOrder);

        bench.run("FDN Block Size " + std::to_string(block_size), [&] {
            uint32_t block_count = kInputSize / block_size;
            for (auto i = 0u; i < block_count; ++i)
            {
                sfFDN::AudioBuffer input_buffer(block_size, 1,
                                                std::span<float>(input).subspan(i * block_size, block_size));
                sfFDN::AudioBuffer output_buffer(block_size, 1,
                                                 std::span<float>(output).subspan(i * block_size, block_size));
                fdn->Process(input_buffer, output_buffer);
            }
        });
    }
}

TEST_CASE("FDNPerf_OrderFFM")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 512;

    constexpr std::array<uint32_t, 5> kStageCount = {2, 3, 4, 5, 6};

    nanobench::Bench bench;
    bench.title("FDN Perf - FFM");
    // bench.minEpochIterations(100);
    bench.batch(kBlockSize);

    for (auto stage_count : kStageCount)
    {
        constexpr uint32_t kFDNOrder = 8;
        auto fdn = CreateFDN(kBlockSize, kFDNOrder);

        auto ffm = CreateFFM(kFDNOrder, stage_count, 1);
        fdn->SetFeedbackMatrix(std::move(ffm));

        std::vector<float> input(kBlockSize, 0.f);
        std::vector<float> output(kBlockSize, 0.f);
        // Fill with white noise
        sfFDN::RNG generator;
        for (auto& i : input)
        {
            i = generator();
        }

        bench.complexityN(stage_count).run("FFM num stages: " + std::to_string(stage_count), [&] {
            sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
            sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
            fdn->Process(input_buffer, output_buffer);
        });
    }

    std::cout << bench.complexityBigO() << "\n";
}
