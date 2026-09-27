// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "nanobench.h"

#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using namespace ankerl;

namespace
{

// The round-trip gain is frequency independent, so scaling the numerator applies it without a separate processor.
sfFDN::FilterCoefficients MakeModeFilter(float frequency, float sample_rate, float loop_gain)
{
    constexpr float kBandwidthHz = 32.F;
    const float radius = 1.F - std::numbers::pi_v<float> * kBandwidthHz / sample_rate;
    const float omega = 2.F * std::numbers::pi_v<float> * frequency / sample_rate;
    const float numerator = loop_gain * (1.F - radius * radius) * 0.5F;

    return {
        .b0 = numerator,
        .b1 = 0.F,
        .b2 = -numerator,
        .a0 = 1.F,
        .a1 = -2.F * radius * std::cos(omega),
        .a2 = radius * radius,
    };
}

std::unique_ptr<sfFDN::FDN> MakePluckedBar(uint32_t mode_count, uint32_t block_size)
{
    constexpr float kSampleRate = 48000.F;
    constexpr float kT60 = 1.5F;

    sfFDN::FDNConfig config;
    config.fdn_size = mode_count;
    config.block_size = block_size;
    config.sample_rate = kSampleRate;
    config.delay_bank_config.block_size = block_size;

    sfFDN::MultichannelProcessorOptions mode_filters;
    mode_filters.channels.reserve(mode_count);
    std::vector<float> excitation(mode_count);

    for (uint32_t mode = 0; mode < mode_count; ++mode)
    {
        const float delay = 256.F + 3.F * static_cast<float>(mode);
        const float frequency = kSampleRate / delay;
        const float loop_gain = std::pow(10.F, -3.F * delay / (kT60 * kSampleRate));
        config.delay_bank_config.delays.push_back(delay);
        excitation[mode] = 1.F / std::sqrt(static_cast<float>(mode + 1U));
        mode_filters.channels.emplace_back(sfFDN::CascadedBiquadsOptions{
            .coeffs = {MakeModeFilter(frequency, kSampleRate, loop_gain)},
        });
    }

    config.input_block_config.parallel_gains_config = {
        .gains = std::move(excitation),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source = sfFDN::GeneratedMatrixOptions{
            .matrix_size = mode_count,
            .generator = sfFDN::ScalarMatrixType::Identity,
        },
    };
    config.loop_filter_configs.emplace_back(std::move(mode_filters));
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(mode_count, 1.F / static_cast<float>(mode_count)),
        .time_varying_config = {},
    };

    return sfFDN::CreateFDNFromConfig(config);
}

} // namespace

TEST_CASE("BandedWaveguide.Perf", "[fdn]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr std::array kModeCounts = {4U, 8U, 16U, 32U, 64U};

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "Banded waveguide mode scaling",
                                                std::chrono::milliseconds(150));
    bench.timeUnit(std::chrono::microseconds(1), "us");
    bench.unit("Process()");

    for (const uint32_t mode_count : kModeCounts)
    {
        auto fdn = MakePluckedBar(mode_count, kBlockSize);
        std::vector<float> input(kBlockSize, 0.F);
        std::vector<float> output(kBlockSize, 0.F);
        input.front() = 1.F;
        const sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);

        fdn->Process(input_buffer, output_buffer);
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        fdn->Clear();

        bench.run("N=" + std::to_string(mode_count), [&] {
            std::ranges::fill(output, 0.F);
            fdn->Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
