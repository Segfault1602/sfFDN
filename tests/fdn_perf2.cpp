#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "sffdn/sffdn.h"

#include "rng.h"

#include "sffdn/types.h"
#include "test_utils.h"

#include <fstream>

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
constexpr float kSampleRate = 48000.f;

std::unique_ptr<sfFDN::FDN> CreateFDN2(uint32_t block_size, uint32_t fdn_order)
{
    sfFDN::AttenuationFilterBankOptions loop_filter_config;
    loop_filter_config.filter_configs.emplace_back(
        sfFDN::TwoBandFilterOptions{.t60s = {1.5f, 0.5f}, .delay = 0.f, .sample_rate = kSampleRate});

    auto config = sfFDN::FDNConfig{.fdn_size = fdn_order,
                                   .transposed = false,
                                   .direct_gain = 0.f,
                                   .block_size = block_size,
                                   .sample_rate = kSampleRate,
                                   .delay_bank_config =
                                       {
                                           .delays = GetDefaultDelays(fdn_order),
                                           .block_size = block_size,
                                       },
                                   .input_block_config =
                                       {
                                           .single_channel_processors = {},
                                           .parallel_gains_config = {.mode = sfFDN::ParallelGainsMode::Split,
                                                                     .gains = std::vector<float>(fdn_order, 0.5f),
                                                                     .time_varying_config = {}},
                                           .multichannel_processors = {},
                                       },
                                   .feedback_matrix_config =
                                       sfFDN::ScalarFeedbackMatrixOptions{
                                           .matrix_size = fdn_order,
                                           .type = sfFDN::ScalarMatrixType::Random,
                                       },
                                   .loop_filter_configs = {loop_filter_config},
                                   .output_block_config =
                                       {
                                           .multichannel_processors = {},
                                           .parallel_gains_config = {.mode = sfFDN::ParallelGainsMode::Merge,
                                                                     .gains = std::vector<float>(fdn_order, 0.5f),
                                                                     .time_varying_config = {}},
                                           .single_channel_processors = {},
                                       },
                                   .tone_correction_filters = {}};

    return CreateFDNFromConfig(config);
}

std::unique_ptr<sfFDN::FDN> CreateFDN_FFM(uint32_t block_size, uint32_t fdn_order, uint32_t stage_count)
{
    sfFDN::CascadedFeedbackMatrixOptions ffm_info = {.matrix_size = fdn_order,
                                                     .stage_count = stage_count,
                                                     .sparsity = 1.f,
                                                     .type = sfFDN::ScalarMatrixType::Random,
                                                     .gain_per_samples = 1.f};

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(ffm_info);

    auto fdn = CreateFDN2(block_size, fdn_order);
    fdn->SetFeedbackMatrix(std::move(ffm));

    return fdn;
}

void RunBenchmark(sfFDN::FDN* fdn, uint32_t block_size, uint32_t fdn_order, nanobench::Bench& bench)
{
    constexpr uint32_t kBufferSize = 16384;

    std::vector<float> input(kBufferSize, 0.f);
    std::vector<float> output(kBufferSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    const uint32_t block_count = 1; // kBufferSize / block_size;
    // if (kBufferSize % block_size != 0)
    // {
    //     throw std::runtime_error("Buffer size must be a multiple of block size");
    // }

    std::string title = std::format("blk={} order={}", block_size, fdn_order);
    bench.batch(block_size);
    bench.run(title, [&] {
        for (auto i = 0u; i < block_count; ++i)
        {
            sfFDN::AudioBuffer input_buffer(block_size, 1, std::span<float>(input).subspan(i * block_size, block_size));
            sfFDN::AudioBuffer output_buffer(block_size, 1,
                                             std::span<float>(output).subspan(i * block_size, block_size));
            fdn->Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output_buffer);
            nanobench::doNotOptimizeAway(input_buffer);
        }
    });
}
} // namespace

TEST_CASE("FDNPerf2_Block", "[FDNPerf2]")
{
    nanobench::Bench bench;
    bench.title("FDN Perf2");
    bench.warmup(100);
    bench.timeUnit(1us, "us");
    bench.minEpochTime(15ms);

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512};
    constexpr std::array kFDNOrders = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 28, 32, 64, 128};

    for (unsigned int block_size : kBlockSizes)
    {
        for (unsigned int fdn_order : kFDNOrders)
        {
            auto fdn = CreateFDN2(block_size, fdn_order);
            RunBenchmark(fdn.get(), block_size, fdn_order, bench);
        }
    }

    std::ofstream csv_file("fdn_perf2_results.csv");
    bench.render(ankerl::nanobench::templates::csv(), csv_file);
    csv_file.close();
}

TEST_CASE("FDNPerf2_Jot", "[FDNPerf2]")
{
    nanobench::Bench bench;
    bench.title("FDN Perf2");
    bench.warmup(100);
    bench.timeUnit(1us, "us");
    bench.minEpochTime(10ms);

    constexpr uint32_t kBlockSize = 64;
    constexpr std::array kFDNOrders = {4, 8, 16, 32, 64};

    for (unsigned int fdn_order : kFDNOrders)
    {
        auto fdn = CreateFDN2(kBlockSize, fdn_order);
        RunBenchmark(fdn.get(), kBlockSize, fdn_order, bench);
    }

    std::ofstream csv_file("fdnperf_jot_results.csv");
    bench.render(ankerl::nanobench::templates::csv(), csv_file);
    csv_file.close();
}

TEST_CASE("FDNPerf2_FFM", "[FDNPerf2]")
{
    nanobench::Bench bench;
    bench.title("FDN Perf2");
    bench.warmup(100);
    bench.timeUnit(1us, "us");
    bench.minEpochTime(10ms);

    constexpr uint32_t kBlockSize = 64;
    constexpr std::array kFDNOrders = {4, 8, 16, 32, 64};

    for (unsigned int fdn_order : kFDNOrders)
    {
        auto fdn = CreateFDN_FFM(kBlockSize, fdn_order, 2);
        RunBenchmark(fdn.get(), kBlockSize, fdn_order, bench);
    }

    std::ofstream csv_file("fdnperf_ffm_results.csv");
    bench.render(ankerl::nanobench::templates::csv(), csv_file);
    csv_file.close();
}