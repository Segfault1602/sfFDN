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

sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions MakeOptions(uint32_t order, bool mixed, bool partial)
{
    const uint32_t stage_count = std::bit_width(order) - 1U;
    sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions options;
    options.matrix.matrix_size = order;
    options.matrix.angles.resize(stage_count, std::numbers::pi_v<float> / 4.0F);
    options.matrix.kernel_types.resize(stage_count, sfFDN::KroneckerKernelType::Rotation);
    options.time_varying_config.resize(stage_count);
    for (uint32_t stage = 0; stage < stage_count; ++stage)
    {
        if (mixed && stage % 2U != 0U)
        {
            options.matrix.kernel_types[stage] = sfFDN::KroneckerKernelType::Reflection;
        }
        options.time_varying_config[stage] = {
            .frequency = (0.5F + (0.1F * static_cast<float>(stage))) / sfFDN::kDefaultSampleRate,
            .amplitude = partial && stage != 0U ? 0.0F : 0.3F,
            .initial_phase = static_cast<float>(stage) / static_cast<float>(stage_count),
        };
    }
    return options;
}

void RunCase(nanobench::Bench& bench, uint32_t order, uint32_t block_size, const char* name,
             const sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions& options)
{
    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix matrix(options);
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);
    bench.run(std::string(name) + " N=" + std::to_string(order) + " B=" + std::to_string(block_size), [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

} // namespace

TEST_CASE("TimeVaryingKroneckerFeedbackMatrixPerf", "[time_varying_kronecker_feedback_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "TimeVaryingKroneckerFeedbackMatrix perf",
                                                std::chrono::milliseconds(500), 2000U);
    sfFDN::test::perf::SetMinEpochIterations(bench, 100'000U);
    for (const uint32_t block_size : sfFDN::test::perf::BlockSizes())
    {
        for (const uint32_t order : sfFDN::test::perf::ChannelCounts())
        {
            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
            RunCase(bench, order, block_size, "modulated rotation", MakeOptions(order, false, false));
            RunCase(bench, order, block_size, "modulated mixed", MakeOptions(order, true, false));
            RunCase(bench, order, block_size, "partially modulated", MakeOptions(order, true, true));
        }
    }
}
