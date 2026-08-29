// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "nanobench.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rng.h"
#include "sffdn/sffdn.h"
#include "sincos.h"

#include "test_utils.h"

using namespace ankerl;

namespace
{

constexpr uint32_t kSampleRate = 48000U;
constexpr float kModulationFrequency = 1.0F / static_cast<float>(kSampleRate);
constexpr float kModulationAmplitude = 0.7F;

std::vector<sfFDN::ModulationOptions> MakeModulationConfig(uint32_t order)
{
    std::vector<sfFDN::ModulationOptions> config(order / 2U);
    for (uint32_t rotation = 0; rotation < config.size(); ++rotation)
    {
        config[rotation] = {
            .frequency = kModulationFrequency,
            .amplitude = kModulationAmplitude,
            .initial_phase = static_cast<float>((rotation * 7U) % order) / static_cast<float>(order),
        };
    }
    return config;
}

void FillRandom(std::span<float> data)
{
    sfFDN::RNG generator(0x9E3779B9U);
    for (float& sample : data)
    {
        sample = generator();
    }
}

void BenchmarkTimeVaryingMatrix(uint32_t order, uint32_t block_size)
{
    std::vector<float> input(order * block_size);
    std::vector<float> time_varying_output(input.size());
    std::vector<float> hadamard_output(input.size());
    std::vector<float> random_output(input.size());
    FillRandom(input);

    sfFDN::TimeVaryingFeedbackMatrix time_varying_hadamard({.matrix_size = order,
                                                            .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
                                                            .time_varying_config = MakeModulationConfig(order)});
    sfFDN::TimeVaryingFeedbackMatrix time_varying_real_schur({.matrix_size = order,
                                                              .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
                                                              .time_varying_config = MakeModulationConfig(order)});
    sfFDN::ScalarFeedbackMatrix hadamard({.matrix_size = order, .type = sfFDN::ScalarMatrixType::Hadamard});
    sfFDN::ScalarFeedbackMatrix random({.matrix_size = order, .type = sfFDN::ScalarMatrixType::Random});
    sfFDN::AudioBuffer input_buffer(block_size, order, input);
    sfFDN::AudioBuffer time_varying_buffer(block_size, order, time_varying_output);
    sfFDN::AudioBuffer hadamard_buffer(block_size, order, hadamard_output);
    sfFDN::AudioBuffer random_buffer(block_size, order, random_output);
    const std::string suffix = " o" + std::to_string(order) + " b" + std::to_string(block_size);

    nanobench::Bench bench;
    bench.title("Time-varying feedback matrix" + suffix);
    bench.timeUnit(std::chrono::microseconds(1), "µs");
    bench.relative(true);
    // No bench.batch(): report time per Process() call (per block), matching FDNPerf.
    bench.minEpochIterations(10000);

    bench.run("Hadamard" + suffix, [&] {
        hadamard.Process(input_buffer, hadamard_buffer);
        nanobench::doNotOptimizeAway(hadamard_output);
    });
    bench.run("TimeVarying Hadamard" + suffix, [&] {
        time_varying_hadamard.Process(input_buffer, time_varying_buffer);
        nanobench::doNotOptimizeAway(time_varying_output);
    });
    bench.run("TimeVarying RealSchur" + suffix, [&] {
        time_varying_real_schur.Process(input_buffer, time_varying_buffer);
        nanobench::doNotOptimizeAway(time_varying_output);
    });
    bench.run("Random" + suffix, [&] {
        random.Process(input_buffer, random_buffer);
        nanobench::doNotOptimizeAway(random_output);
    });
}

void BenchmarkFDN(sfFDN::FDN& fdn, std::string_view name, nanobench::Bench& bench)
{
    constexpr uint32_t kBlockSize = 128U;
    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    FillRandom(input);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1U, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1U, output);
    bench.run(std::string(name), [&] {
        fdn.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

} // namespace

TEST_CASE("TimeVaryingFeedbackMatrixPerf_MatrixSweep", "[TimeVaryingFeedbackMatrix][perf]")
{
    for (const uint32_t order : {8U, 16U, 32U})
    {
        for (const uint32_t block_size : {64U, 128U, 256U})
        {
            BenchmarkTimeVaryingMatrix(order, block_size);
        }
    }
}

TEST_CASE("TimeVaryingFeedbackMatrixPerf_SinCosUnit", "[TimeVaryingFeedbackMatrix][perf]")
{
    constexpr std::array kAngles = {
        -std::numbers::pi_v<float>,         -0.7F * std::numbers::pi_v<float>,
        -0.25F * std::numbers::pi_v<float>, 0.0F,
        0.25F * std::numbers::pi_v<float>,  0.7F * std::numbers::pi_v<float>,
        std::numbers::pi_v<float>,          1.75F * std::numbers::pi_v<float>,
    };
    float sine = 0.0F;
    float cosine = 0.0F;

    nanobench::Bench bench;
    bench.title("SinCosUnit");
    bench.timeUnit(std::chrono::nanoseconds(1), "ns");
    bench.batch(kAngles.size());
    bench.minEpochIterations(1000000);
    bench.run("SinCosUnit representative angles", [&] {
        float output_sum = 0.0F;
        for (const float angle : kAngles)
        {
            sfFDN::SinCosUnit(angle, sine, cosine);
            output_sum += sine + cosine;
        }
        nanobench::doNotOptimizeAway(output_sum);
    });
}

TEST_CASE("TimeVaryingFeedbackMatrixPerf_FDN", "[TimeVaryingFeedbackMatrix][perf]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kOrder = 16U;

    auto static_fdn = CreateFDN(kBlockSize, kOrder);
    static_fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(
        sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = kOrder, .type = sfFDN::ScalarMatrixType::Hadamard}));

    auto time_varying_fdn = CreateFDN(kBlockSize, kOrder);
    time_varying_fdn->SetFeedbackMatrix(
        std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(sfFDN::TimeVaryingFeedbackMatrixOptions{
            .matrix_size = kOrder, .time_varying_config = MakeModulationConfig(kOrder)}));

    auto time_varying_fdn_schur = CreateFDN(kBlockSize, kOrder);
    time_varying_fdn_schur->SetFeedbackMatrix(std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(
        sfFDN::TimeVaryingFeedbackMatrixOptions{.matrix_size = kOrder,
                                                .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
                                                .time_varying_config = MakeModulationConfig(kOrder)}));

    nanobench::Bench bench;
    bench.title("FDN feedback matrix comparison o16 b128");
    bench.timeUnit(std::chrono::microseconds(1), "µs");
    bench.relative(true);
    // No bench.batch(): report time per Process() call (per block), matching FDNPerf.
    bench.minEpochIterations(10000);

    BenchmarkFDN(*static_fdn, "FDN Hadamard o16 b128", bench);
    BenchmarkFDN(*time_varying_fdn, "FDN TimeVarying o16 b128", bench);
    BenchmarkFDN(*time_varying_fdn_schur, "FDN TimeVarying Schur o16 b128", bench);
}
