#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "sffdn/oscillator.h"

TEST_CASE("SineWave Generate produces sine samples", "[oscillator]")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kSampleRate = 48000;
    constexpr float kFrequency = 1000.0f; // A4 note

    sfFDN::SineWave sine_wave(kFrequency / kSampleRate);

    constexpr uint32_t kOutputSize = 1 << 10;
    std::vector<float> output(kOutputSize, 0.f);

    constexpr uint32_t kBlockCount = kOutputSize / kBlockSize;
    for (auto i = 0u; i < kBlockCount; ++i)
    {
        auto block_span = std::span(output).subspan(i * kBlockSize, kBlockSize);
        sine_wave.Generate(block_span);
    }

    constexpr float kPhaseIncrement = kFrequency / kSampleRate;
    float phase = 0;
    for (auto i = 0u; i < kOutputSize; ++i)
    {
        float expected_value = std::sinf(phase * 2.0f * std::numbers::pi);
        phase += kPhaseIncrement;
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_value, 7e-4));
    }
}

TEST_CASE("SineWave sample APIs agree", "[oscillator]")
{
    constexpr uint32_t kSize = 17;
    constexpr float kFrequency = 0.13f;
    constexpr float kPhase = 0.37f;
    sfFDN::SineWave next_out(kFrequency, kPhase);
    sfFDN::SineWave tick(kFrequency, kPhase);
    sfFDN::SineWave generate(kFrequency, kPhase);
    std::array<float, kSize> generated{};
    generate.Generate(generated);

    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(next_out.NextOut(), Catch::Matchers::WithinAbs(generated[i], 7e-4f));
        REQUIRE_THAT(tick.Tick(), Catch::Matchers::WithinAbs(generated[i], 7e-4f));
        next_out.Tick();
    }
}

TEST_CASE("SineWave applies controls and wraps normalized frequency", "[oscillator]")
{
    constexpr float kFrequency = 0.75f;
    constexpr float kInitialPhase = 0.25f;
    constexpr float kPhaseOffset = 0.125f;
    constexpr float kAmplitude = 0.5f;
    constexpr float kOffset = -0.25f;
    constexpr uint32_t kSize = 5;

    sfFDN::SineWave sine_wave(kFrequency, kInitialPhase);
    sine_wave.SetPhaseOffset(kPhaseOffset);
    sine_wave.SetAmplitude(kAmplitude);
    sine_wave.SetOffset(kOffset);
    REQUIRE(sine_wave.GetFrequency() == kFrequency);
    REQUIRE(sine_wave.GetPhaseOffset() == kPhaseOffset);
    REQUIRE(sine_wave.GetAmplitude() == kAmplitude);
    REQUIRE(sine_wave.GetOffset() == kOffset);

    for (auto i = 0u; i < kSize; ++i)
    {
        const float phase = kInitialPhase + (static_cast<float>(i) * kFrequency) + kPhaseOffset;
        const float expected = (std::sin(phase * 2.f * std::numbers::pi_v<float>) * kAmplitude) + kOffset;
        REQUIRE_THAT(sine_wave.Tick(), Catch::Matchers::WithinAbs(expected, 7e-4f));
    }

    sine_wave.ResetPhase();
    REQUIRE_THAT(sine_wave.NextOut(),
                 Catch::Matchers::WithinAbs(
                     (std::sin(kPhaseOffset * 2.f * std::numbers::pi_v<float>) * kAmplitude) + kOffset, 7e-4f));
}

TEST_CASE("SineWave safely wraps phases that round to the table endpoint", "[oscillator]")
{
    constexpr float kPhaseOffset = -0.3f;
    constexpr float kFrequency = 0.01f;
    const float initial_phase = std::nextafter(0.3f, 0.f);
    std::array<float, 9> output{};

    sfFDN::SineWave sine_wave(kFrequency, initial_phase);
    sine_wave.SetPhaseOffset(kPhaseOffset);
    sine_wave.Generate(output);

    float phase = initial_phase;
    for (const float sample : output)
    {
        const float expected = std::sin((phase + kPhaseOffset) * 2.f * std::numbers::pi_v<float>);
        REQUIRE_THAT(sample, Catch::Matchers::WithinAbs(expected, 7e-4f));
        phase += kFrequency;
    }
}

TEST_CASE("SineWave Multiply matches scalar modulation and accumulation", "[oscillator]")
{
    constexpr std::array<float, 11> kInput = {1.f, -2.f, 0.5f, -0.25f, 4.f, 3.f, -1.f, 0.75f, -0.5f, 2.f, -3.f};
    constexpr float kFrequency = 0.2f;
    constexpr float kPhase = 0.1f;
    constexpr float kPhaseOffset = 0.3f;
    constexpr float kAmplitude = 0.75f;
    constexpr float kOffset = -0.125f;
    std::array<float, kInput.size()> multiplied{};
    std::array<float, kInput.size()> accumulated = {2.f, -1.f, 0.5f, 3.f, -2.f, 1.f, 4.f, -0.75f, 1.5f, 0.f, 2.5f};
    const auto original_accumulated = accumulated;

    sfFDN::SineWave multiply(kFrequency, kPhase);
    sfFDN::SineWave multiply_accumulate(kFrequency, kPhase);
    for (auto* oscillator : {&multiply, &multiply_accumulate})
    {
        oscillator->SetPhaseOffset(kPhaseOffset);
        oscillator->SetAmplitude(kAmplitude);
        oscillator->SetOffset(kOffset);
    }

    multiply.Multiply(kInput, multiplied);
    multiply_accumulate.MultiplyAccumulate(kInput, accumulated);

    for (auto i = 0u; i < kInput.size(); ++i)
    {
        const float phase = kPhase + (static_cast<float>(i) * kFrequency) + kPhaseOffset;
        const float oscillator = (std::sin(phase * 2.f * std::numbers::pi_v<float>) * kAmplitude) + kOffset;
        const float expected = kInput[i] * oscillator;
        REQUIRE_THAT(multiplied[i], Catch::Matchers::WithinAbs(expected, 7e-4f));
        REQUIRE_THAT(accumulated[i], Catch::Matchers::WithinAbs(original_accumulated[i] + expected, 7e-4f));
    }
}