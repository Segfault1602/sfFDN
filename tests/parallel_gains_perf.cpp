#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "sffdn/sffdn.h"

#include "rng.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("ParallelGainsPerf", "[Gains]")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kChannelCount = 16;
    constexpr std::array kGains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize * kChannelCount, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    nanobench::Bench bench;
    bench.title("Parallel Gain Perf");
    bench.warmup(100);

    bench.minEpochIterations(100000);
    sfFDN::ParallelGains input_gains(sfFDN::ParallelGainsMode::Split);
    input_gains.SetGains(kGains);
    bench.run("ParallelGains - Input", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);
        input_gains.Process(input_buffer, output_buffer);
    });

    sfFDN::ParallelGains output_gains(sfFDN::ParallelGainsMode::Merge);
    output_gains.SetGains(kGains);
    bench.run("ParallelGains - Output", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, output);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, input);
        output_gains.Process(input_buffer, output_buffer);
    });

    constexpr std::array kFreqs = {0.5f / kSampleRate, 1.0f / kSampleRate, 1.5f / kSampleRate, 2.0f / kSampleRate,
                                   2.5f / kSampleRate, 3.0f / kSampleRate, 3.5f / kSampleRate, 4.0f / kSampleRate,
                                   4.5f / kSampleRate, 5.0f / kSampleRate, 5.5f / kSampleRate, 6.0f / kSampleRate,
                                   6.5f / kSampleRate, 7.0f / kSampleRate, 7.5f / kSampleRate, 8.0f / kSampleRate};

    sfFDN::TimeVaryingParallelGains tv_input_gains(sfFDN::ParallelGainsMode::Split);
    tv_input_gains.SetCenterGains(kGains);
    tv_input_gains.SetLfoFrequency(kFreqs);

    bench.minEpochIterations(5000);
    bench.run("Time Varying ParallelGains - Input", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);
        tv_input_gains.Process(input_buffer, output_buffer);
    });

    sfFDN::TimeVaryingParallelGains tv_output_gains(sfFDN::ParallelGainsMode::Merge);
    tv_output_gains.SetCenterGains(kGains);
    tv_output_gains.SetLfoFrequency(kFreqs);

    bench.run("Time Varying ParallelGains - Output", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, output);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, input);
        tv_output_gains.Process(input_buffer, output_buffer);
    });
}