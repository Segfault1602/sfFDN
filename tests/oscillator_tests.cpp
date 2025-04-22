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
#include "test_utils.h"

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

    WriteWavFile("sine_wave.wav", output);

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

    WriteWavFile("rng_noise.wav", output);
}