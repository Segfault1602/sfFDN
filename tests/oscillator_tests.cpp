#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <sndfile.h>
#include <sys/types.h>
#include <vector>

#include "rng.h"
#include "sffdn/sffdn.h"

#include "sffdn/oscillator.h"

TEST_CASE("SineWave")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kSampleRate = 48000;
    constexpr float kFrequency = 1000.0f; // A4 note

    sfFDN::SineWave sine_wave(kFrequency / kSampleRate);

    constexpr uint32_t kOutputSize = 1 << 10;
    std::vector<float> output(kOutputSize, 0.f); // Two channels for stereo output

    constexpr uint32_t kBlockCount = kOutputSize / kBlockSize;
    for (auto i = 0u; i < kBlockCount; ++i)
    {
        auto block_span = std::span(output).subspan(i * kBlockSize, kBlockSize);
        sine_wave.Generate(block_span);
    }

    SF_INFO sf_info;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sf_info.channels = 1;
    sf_info.samplerate = kSampleRate;
    SNDFILE* file = sf_open("sine_wave.wav", SFM_WRITE, &sf_info);
    if (file == nullptr)
    {
        std::print("Error opening file for writing: {}\n", sf_strerror(file));
        return;
    }

    sf_writef_float(file, output.data(), kOutputSize);
    sf_close(file);

    constexpr float kPhaseIncrement = kFrequency / kSampleRate;
    float phase = 0;
    for (auto i = 0u; i < kOutputSize; ++i)
    {
        float expected_value = std::sinf(phase * 2.0f * std::numbers::pi);
        phase += kPhaseIncrement;
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_value, 7e-4));
    }
}

TEST_CASE("Noise")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kOutputSize = kSampleRate;

    std::vector<float> output(kOutputSize, 0.f);
    sfFDN::RNG rng;
    for (auto i = 0u; i < kOutputSize; ++i)
    {
        output[i] = rng();
    }

    SF_INFO sf_info;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sf_info.channels = 1;
    sf_info.samplerate = kSampleRate;
    SNDFILE* file = sf_open("rng_noise.wav", SFM_WRITE, &sf_info);
    if (file == nullptr)
    {
        std::print("Error opening file for writing: {}\n", sf_strerror(file));
        return;
    }

    sf_writef_float(file, output.data(), kOutputSize);
    sf_close(file);
}