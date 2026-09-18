#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
enum class FDNFamily : uint8_t
{
    HouseholderElevenStage,
    RandomTwoBand,
};

struct FamilyInfo
{
    FDNFamily family;
    std::string_view name;
};

constexpr std::array kFamilies = {
    FamilyInfo{.family = FDNFamily::HouseholderElevenStage, .name = "Householder 11-stage"},
    FamilyInfo{.family = FDNFamily::RandomTwoBand, .name = "Random two-band"},
};

std::vector<float> MakeDelays(uint32_t order, uint32_t block_size)
{
    // A fixed seed is mandatory here: with the default seed of 0, GetDelayLengths draws from std::random_device, so
    // every process run would get different delay lengths and therefore different delay-line working sets. That makes
    // any before/after comparison across runs meaningless.
    constexpr uint32_t kDelaySeed = 0x5F4E3D2CU;
    const uint32_t minimum_delay = std::max(512U, block_size + 64U);
    return sfFDN::GetDelayLengths(order, minimum_delay, minimum_delay + 8192U, sfFDN::DelayLengthType::Uniform,
                                  kDelaySeed);
}

std::unique_ptr<sfFDN::FDN> MakeFDN(FDNFamily family, uint32_t block_size, uint32_t order)
{
    auto fdn = std::make_unique<sfFDN::FDN>(order, block_size, false);
    const std::vector<float> gains(order, 0.5F);
    const std::vector<float> delays = MakeDelays(order, block_size);
    REQUIRE(fdn->SetInputGains(gains));
    REQUIRE(fdn->SetOutputGains(gains));
    REQUIRE(fdn->SetDelays(delays));
    fdn->SetDirectGain(0.F);

    if (family == FDNFamily::HouseholderElevenStage)
    {
        REQUIRE(fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(sfFDN::ScalarFeedbackMatrixOptions{
            .source =
                sfFDN::GeneratedMatrixOptions{
                    .matrix_size = order,
                    .generator = sfFDN::ScalarMatrixType::Householder,
                },
        })));
        REQUIRE(fdn->SetLoopFilter(GetLoopFilter(order, 11U)));
        REQUIRE(fdn->SetTCFilter(GetDefaultTCFilter()));
        return fdn;
    }

    REQUIRE(fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = order,
                .generator = sfFDN::ScalarMatrixType::Random,
                .rng_seed = 4242U,
            },
    })));
    const sfFDN::FilterDesigner designer(static_cast<float>(sfFDN::kDefaultSampleRate));
    REQUIRE(fdn->SetLoopFilter(sfFDN::CreateAttenuationFilterBank(
        sfFDN::TwoBandFilterOptions{
            .t60s = {1.5F, 0.5F},
            .delay = 0.F,
        },
        delays, designer)));
    return fdn;
}

void RunFDNBenchmark(const FamilyInfo& family, uint32_t order, uint32_t block_size, nanobench::Bench& bench)
{
    auto fdn = MakeFDN(family.family, block_size, order);
    std::vector<float> input(block_size);
    std::vector<float> output(block_size);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    std::ranges::fill(output, 0.F);
    fdn->Process(input_buffer, output_buffer);
    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    fdn->Clear();

    bench.run(std::string(family.name) + " N=" + std::to_string(order) + " B=" + std::to_string(block_size), [&] {
        std::ranges::fill(output, 0.F);
        fdn->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("FDNPerf", "[fdn]")
{
    nanobench::Bench bench;
    // A whole FDN block is tens of microseconds, so the default 10 ms epoch only buys a couple of hundred iterations
    // and nanobench flags the eleven-stage row as unstable at over 20% err. A longer epoch is what makes this row
    // usable for before/after comparison.
    sfFDN::test::perf::ConfigureThroughputBench(bench, "FDN perf", std::chrono::milliseconds(150));

    for (const FamilyInfo& family : kFamilies)
    {
        for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
        {
            for (const uint32_t order : sfFDN::test::perf::ChannelCounts())
            {
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
                RunFDNBenchmark(family, order, block_size, bench);
            }
        }
    }
}

TEST_CASE("FDNPerf_BigO", "[fdn][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;

    for (const FamilyInfo& family : kFamilies)
    {
        nanobench::Bench bench;
        sfFDN::test::perf::ConfigureComplexityBench(
            bench, "FDN " + std::string(family.name) + " B=" + std::to_string(kBlockSize));

        for (const uint32_t order : sfFDN::test::perf::kExtendedChannelCounts)
        {
            bench.complexityN(order);
            RunFDNBenchmark(family, order, kBlockSize, bench);
        }
        std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
    }
}

TEST_CASE("FDNPerf_FIR", "[fdn]")
{
    constexpr uint32_t kOrder = 16U;
    const std::vector<float> fir = ReadWavFile("./tests/data/att_fir_1153.wav");
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "FDN FIR-loop perf");

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        auto fdn = MakeFDN(FDNFamily::HouseholderElevenStage, block_size, kOrder);
        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (uint32_t channel = 0; channel < kOrder; ++channel)
        {
            filter_bank->AddFilter(std::make_unique<sfFDN::PartitionedConvolver>(block_size, fir));
        }
        REQUIRE(fdn->SetLoopFilter(std::move(filter_bank)));

        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);
        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);

        std::ranges::fill(output, 0.F);
        fdn->Process(input_buffer, output_buffer);
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        fdn->Clear();

        bench.run("N=16 B=" + std::to_string(block_size), [&] {
            std::ranges::fill(output, 0.F);
            fdn->Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("FDNPerf_FFM", "[fdn]")
{
    constexpr uint32_t kOrder = 16U;
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "FDN FilterFeedbackMatrix perf");

    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        auto fdn = MakeFDN(FDNFamily::HouseholderElevenStage, block_size, kOrder);
        REQUIRE(
            fdn->SetFeedbackMatrix(std::make_unique<sfFDN::FilterFeedbackMatrix>(sfFDN::CascadedFeedbackMatrixOptions{
                .matrix_size = kOrder,
                .stage_count = 2U,
                .sparsity = 1.F,
                .generator = sfFDN::ScalarMatrixType::Hadamard,
                .gain_per_samples = 1.F,
            })));

        std::vector<float> input(block_size);
        std::vector<float> output(block_size);
        sfFDN::test::perf::FillNoise(input);
        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);

        std::ranges::fill(output, 0.F);
        fdn->Process(input_buffer, output_buffer);
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        fdn->Clear();

        bench.run("N=16 B=" + std::to_string(block_size), [&] {
            std::ranges::fill(output, 0.F);
            fdn->Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
