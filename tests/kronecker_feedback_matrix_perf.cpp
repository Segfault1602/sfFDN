// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "nanobench.h"

#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <bit>
#include <chrono>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{

sfFDN::KroneckerFeedbackMatrixOptions MakeOptions(uint32_t order, bool reflections, bool mixed)
{
    const uint32_t stage_count = std::bit_width(order) - 1U;
    sfFDN::KroneckerFeedbackMatrixOptions options;
    options.matrix_size = order;
    options.angles.resize(stage_count, std::numbers::pi_v<float> / 4.0F);
    options.kernel_types.resize(stage_count, reflections ? sfFDN::KroneckerKernelType::Reflection
                                                         : sfFDN::KroneckerKernelType::Rotation);
    if (mixed)
    {
        for (uint32_t stage = 0; stage < stage_count; ++stage)
        {
            options.kernel_types[stage] =
                stage % 2U == 0U ? sfFDN::KroneckerKernelType::Rotation : sfFDN::KroneckerKernelType::Reflection;
        }
    }
    return options;
}

void RunCase(nanobench::Bench& bench, uint32_t order, uint32_t block_size, const char* name,
             const sfFDN::KroneckerFeedbackMatrixOptions& options)
{
    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    sfFDN::KroneckerFeedbackMatrix matrix(options);
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);
    bench.run(std::string(name) + " N=" + std::to_string(order) + " B=" + std::to_string(block_size), [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

} // namespace

TEST_CASE("KroneckerFeedbackMatrixPerf", "[kronecker_feedback_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "KroneckerFeedbackMatrix perf", std::chrono::milliseconds(500),
                                                2000U);
    sfFDN::test::perf::SetMinEpochIterations(bench, 100'000U);
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t order : sfFDN::test::perf::ChannelCounts())
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
            RunCase(bench, order, block_size, "static rotation", MakeOptions(order, false, false));
            RunCase(bench, order, block_size, "static reflection", MakeOptions(order, true, false));
            RunCase(bench, order, block_size, "static mixed", MakeOptions(order, false, true));
        }
    }
}

TEST_CASE("KroneckerFeedbackMatrixPerf_AngleUpdateStrategies", "[kronecker_feedback_matrix][.diagnostic]")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kStages = 3U;
    std::vector<float> input(kOrder * kBlockSize);
    std::vector<float> streamed_output(input.size());
    std::vector<float> sampled_output(input.size());
    sfFDN::test::perf::FillNoise(input);
    std::vector<float> offsets(kStages * kBlockSize);
    std::vector<float> absolute_angles(kStages * kBlockSize);
    for (uint32_t stage = 0; stage < kStages; ++stage)
    {
        for (uint32_t sample = 0; sample < kBlockSize; ++sample)
        {
            const float offset =
                0.3F * std::numbers::pi_v<float> *
                std::sin((2.0F * std::numbers::pi_v<float> * static_cast<float>((stage + 1U) * sample)) /
                         static_cast<float>(kBlockSize));
            offsets[(stage * kBlockSize) + sample] = offset;
            absolute_angles[(stage * kBlockSize) + sample] = (std::numbers::pi_v<float> / 4.0F) + offset;
        }
    }

    sfFDN::KroneckerFeedbackMatrix streamed({.matrix_size = kOrder, .angles = {}, .kernel_types = {}});
    sfFDN::KroneckerFeedbackMatrix sampled({.matrix_size = kOrder, .angles = {}, .kernel_types = {}});
    const sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer streamed_buffer(kBlockSize, kOrder, streamed_output);
    sfFDN::AudioBuffer sampled_buffer(kBlockSize, kOrder, sampled_output);

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureComplexityBench(bench, "Kronecker angle update strategies",
                                               std::chrono::milliseconds(200), 2000U);
    sfFDN::test::perf::SetMinEpochIterations(bench, 100'000U);
    bench.run("block angle offsets", [&] {
        streamed.ProcessWithAngleOffsets(input_buffer, streamed_buffer, offsets, 0b111U);
        nanobench::doNotOptimizeAway(streamed_output);
    });
    bench.run("sample SetAngles plus Process", [&] {
        std::array<float, kStages> angles{};
        for (uint32_t sample = 0; sample < kBlockSize; ++sample)
        {
            for (uint32_t stage = 0; stage < kStages; ++stage)
            {
                angles[stage] = absolute_angles[(stage * kBlockSize) + sample];
            }
            sampled.SetAngles(angles);
            const auto input_sample = input_buffer.Offset(sample, 1U);
            auto output_sample = sampled_buffer.Offset(sample, 1U);
            sampled.Process(input_sample, output_sample);
        }
        nanobench::doNotOptimizeAway(sampled_output);
    });
}
