#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "sffdn/parallel_gains.h"
#include "sffdn/sffdn.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("ParallelGainsPerf", "[Gains]")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t N = 16;
    constexpr std::array kGains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("Parallel Gain Perf");
    bench.warmup(100);

    bench.minEpochIterations(100000);
    sfFDN::ParallelGains input_gains(sfFDN::ParallelGainsMode::Multiplexed);
    input_gains.SetGains(kGains);
    bench.run("ParallelGains - Input", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        input_gains.Process(input_buffer, output_buffer);
    });

    sfFDN::ParallelGains output_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    output_gains.SetGains(kGains);
    bench.run("ParallelGains - Output", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, output);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, input);
        output_gains.Process(input_buffer, output_buffer);
    });

    constexpr std::array kFreqs = {0.5f / SR, 1.0f / SR, 1.5f / SR, 2.0f / SR, 2.5f / SR, 3.0f / SR,
                                   3.5f / SR, 4.0f / SR, 4.5f / SR, 5.0f / SR, 5.5f / SR, 6.0f / SR,
                                   6.5f / SR, 7.0f / SR, 7.5f / SR, 8.0f / SR};

    sfFDN::TimeVaryingParallelGains tv_input_gains(sfFDN::ParallelGainsMode::Multiplexed);
    tv_input_gains.SetCenterGains(kGains);
    tv_input_gains.SetLfoFrequency(kFreqs);

    bench.minEpochIterations(5000);
    bench.run("Time Varying ParallelGains - Input", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
        tv_input_gains.Process(input_buffer, output_buffer);
    });

    sfFDN::TimeVaryingParallelGains tv_output_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    tv_output_gains.SetCenterGains(kGains);
    tv_output_gains.SetLfoFrequency(kFreqs);

    bench.run("Time Varying ParallelGains - Output", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, N, output);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, input);
        tv_output_gains.Process(input_buffer, output_buffer);
    });
}