// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "nanobench.h"

#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ankerl;

namespace
{
struct ModeInfo
{
    sfFDN::TimeVaryingMatrixMode mode;
    std::string_view name;
};

constexpr std::array kModes = {
    ModeInfo{sfFDN::TimeVaryingMatrixMode::Hadamard, "Hadamard"},
    ModeInfo{sfFDN::TimeVaryingMatrixMode::RealSchur, "RealSchur"},
};

static_assert(kModes.size() == std::to_underlying(sfFDN::TimeVaryingMatrixMode::Count));

std::vector<sfFDN::ModulationOptions> MakeModulationConfig(uint32_t order)
{
    std::vector<sfFDN::ModulationOptions> config(order / 2U);
    for (uint32_t rotation = 0; rotation < config.size(); ++rotation)
    {
        config[rotation] = {
            .frequency = (0.75F + (0.05F * static_cast<float>(rotation))) /
                         static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 0.7F,
            .initial_phase = static_cast<float>((rotation * 7U) % order) / static_cast<float>(order),
        };
    }
    return config;
}

void RunTimeVaryingFeedbackMatrixBenchmark(const ModeInfo& mode, uint32_t order, uint32_t block_size,
                                           nanobench::Bench& bench)
{
    std::vector<float> input(static_cast<size_t>(order) * block_size);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);

    sfFDN::TimeVaryingFeedbackMatrix matrix({
        .matrix_size = order,
        .mode = mode.mode,
        .time_varying_config = MakeModulationConfig(order),
        .rng_seed = 4242U,
    });
    const sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer output_buffer(block_size, order, output);
    const std::string name =
        std::string(mode.name) + " N=" + std::to_string(order) + " B=" + std::to_string(block_size);

    bench.run(name, [&] {
        matrix.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("TimeVaryingFeedbackMatrixPerf", "[time_varying_matrix]")
{
    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "TimeVaryingFeedbackMatrix perf");

    for (const ModeInfo& mode : kModes)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            for (const uint32_t order : sfFDN::test::perf::kChannelCounts)
            {
                const bool needs_iteration_floor =
                    mode.mode == sfFDN::TimeVaryingMatrixMode::Hadamard && order == 32U;
                bench.minEpochIterations(needs_iteration_floor ? 10'000U : 1U);
                sfFDN::test::perf::SetChannelSampleBatch(bench, block_size, order);
                RunTimeVaryingFeedbackMatrixBenchmark(mode, order, block_size, bench);
            }
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrixPerf_BigO", "[time_varying_matrix][.diagnostic]")
{
    constexpr uint32_t kBlockSize = 128U;

    for (const ModeInfo& mode : kModes)
    {
        nanobench::Bench bench;
        sfFDN::test::perf::ConfigureComplexityBench(
            bench, "TimeVaryingFeedbackMatrix " + std::string(mode.name) + " B=" + std::to_string(kBlockSize));

        for (const uint32_t order : sfFDN::test::perf::kChannelCounts)
        {
            bench.complexityN(order);
            RunTimeVaryingFeedbackMatrixBenchmark(mode, order, kBlockSize, bench);
        }
        std::cout << sfFDN::test::perf::FormatComplexityFits(bench.complexityBigO()) << '\n';
    }
}
