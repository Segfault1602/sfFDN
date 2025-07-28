#include "doctest.h"

#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <sndfile.h>
#include <vector>

#include "audio_buffer.h"

#include "oscillator.h"

TEST_CASE("SineWave")
{
    constexpr size_t kBlockSize = 128;
    constexpr float kFrequency = 1000.0f; // A4 note
    constexpr uint32_t kSampleRate = 48000;

    sfFDN::SineWave sine_wave(kFrequency, kSampleRate);

    constexpr size_t kOutputSize = 1 << 10;
    std::vector<float> output(kOutputSize, 0.f); // Two channels for stereo output

    const uint32_t kBlockCount = kOutputSize / kBlockSize;
    for (auto i = 0; i < kBlockCount; ++i)
    {
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data() + i * kBlockSize);
        sine_wave.Generate(output_buffer);
    }

    SF_INFO sf_info;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sf_info.channels = 1;
    sf_info.samplerate = kSampleRate;
    SNDFILE* file = sf_open("sine_wave.wav", SFM_WRITE, &sf_info);
    if (!file)
    {
        std::print("Error opening file for writing: {}\n", sf_strerror(file));
        return;
    }

    sf_writef_float(file, output.data(), kOutputSize);
    sf_close(file);

    const float kPhaseIncrement = (2.0f * std::numbers::pi * kFrequency) / kSampleRate;
    for (auto i = 0; i < kOutputSize; ++i)
    {
        float expected_value = std::sinf(kPhaseIncrement * i);
        CHECK(output[i] == doctest::Approx(expected_value).epsilon(1e-4));
    }
}