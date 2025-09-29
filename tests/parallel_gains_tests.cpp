#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include <sndfile.h>

#include "sffdn/parallel_gains.h"
#include "sffdn/sffdn.h"

TEST_CASE("ParallelGainsInput")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.25f, 0.5f, 0.75f, 1.f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Multiplexed);
    parallel_gains.SetGains(kGains);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kChannelCount * kBlockSize, 0.f);
    for (auto i = 0u; i < kBlockSize; ++i)
    {
        input[i] = i;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 0.25, 0.5, 0.75, 1,   1.25, 1.5, 1.75, 2,   2.25, 0,   0.5,  1,   1.5,
                                       2, 2.5,  3,   3.5,  4,   4.5,  0,   0.75, 1.5, 2.25, 3,   3.75, 4.5, 5.25,
                                       6, 6.75, 0,   1.f,  2.f, 3.f,  4.f, 5.f,  6.f, 7.f,  8.f, 9.f};

    REQUIRE(output.size() == expected_out.size());
    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("ParallelGainsOutput")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.5f, 0.5f, 0.5f, 0.5f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    parallel_gains.SetGains(kGains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);

    for (auto i = 0u; i < kChannelCount; ++i)
    {
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            input[i * kBlockSize + j] = j;
        }
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    REQUIRE(output.size() == expected_out.size());

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

// With frequency and amplitude to 0, this should behave the same as a normal ParallelGain
TEST_CASE("TimeVaryingParallelGainsInput_static")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.25f, 0.5f, 0.75f, 1.f};
    sfFDN::TimeVaryingParallelGains tv_parallel_gains(sfFDN::ParallelGainsMode::Multiplexed);
    tv_parallel_gains.SetCenterGains(kGains);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = i;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    tv_parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out(kChannelCount * kBlockSize, 0.f);
    sfFDN::AudioBuffer expected_out_buffer(kBlockSize, kChannelCount, expected_out);
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Multiplexed);
    parallel_gains.SetGains(kGains);
    parallel_gains.Process(input_buffer, expected_out_buffer);

    REQUIRE(output.size() == expected_out.size());
    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("TimeVaryingParallelGainsOutput_static")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.5f, 0.5f, 0.5f, 0.5f};
    sfFDN::TimeVaryingParallelGains tv_parallel_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    tv_parallel_gains.SetCenterGains(kGains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    for (auto channel : input_buffer)
    {
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            channel[j] = j;
        }
    }

    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
    tv_parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out(kBlockSize, 0.f);
    sfFDN::AudioBuffer expected_out_buffer(kBlockSize, 1, expected_out);
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    parallel_gains.SetGains(kGains);
    parallel_gains.Process(input_buffer, expected_out_buffer);

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("TimeVaryingParallelGainsInput")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = kSampleRate;
    constexpr std::array<float, kChannelCount> kCenterGains = {0.25f, 0.5f, -0.0f, -0.5f};
    constexpr std::array<float, kChannelCount> kLfoRates = {1.f / kSampleRate, 2.f / kSampleRate, 3.f / kSampleRate,
                                                            4.f / kSampleRate};
    constexpr std::array<float, kChannelCount> kLfoAmps = {0.25f, 0.33f, 0.2f, -0.1f};

    sfFDN::TimeVaryingParallelGains tv_parallel_gains(sfFDN::ParallelGainsMode::Multiplexed);
    tv_parallel_gains.SetCenterGains(kCenterGains);
    tv_parallel_gains.SetLfoFrequency(kLfoRates);
    tv_parallel_gains.SetLfoAmplitude(kLfoAmps);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    std::ranges::fill(input, 1.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    tv_parallel_gains.Process(input_buffer, output_buffer);

    SF_INFO sf_info;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sf_info.channels = kChannelCount;
    sf_info.samplerate = kSampleRate;

    SNDFILE* file = sf_open("tv_gains.wav", SFM_WRITE, &sf_info);
    if (file == nullptr)
    {
        std::cerr << "Error opening file for writing: " << sf_strerror(file) << "\n";
        return;
    }

    for (auto i = 0u; i < kBlockSize; ++i)
    {
        std::array<float, kChannelCount> frame{};
        for (auto j = 0u; j < kChannelCount; ++j)
        {
            frame[j] = output_buffer.GetChannelSpan(j)[i];
        }
        sf_writef_float(file, frame.data(), 1);
    }

    sf_close(file);
}