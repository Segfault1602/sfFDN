#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include "sffdn/sffdn.h"

#include "oscillator.h"

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
void std_sin(sfFDN::AudioBuffer& output, float frequency, uint32_t sample_rate)
{
    float phase = 0;
    float phase_increment = frequency / sample_rate;
    auto first_channel = output.GetChannelSpan(0);
    for (float& i : first_channel)
    {
        i = std::sinf(phase * 2.0f * std::numbers::pi);
        phase += phase_increment;
    }
}

} // namespace

TEST_CASE("SineWave")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 64;

    sfFDN::SineWave sine_wave(10.0f, kSampleRate);

    nanobench::Bench bench;
    bench.title("SineWave");
    bench.minEpochIterations(500000);
    bench.relative(true);
    bench.batch(kBlockSize);
    // bench.timeUnit(1us, "us");

    std::vector<float> output(kBlockSize, 0.f);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    bench.run("sfFDN::Oscillator", [&]() {
        sine_wave.Generate(output_buffer);
        nanobench::doNotOptimizeAway(output_buffer);
    });

    bench.run("std::sinf", [&]() {
        std_sin(output_buffer, 10.0f, kSampleRate);
        nanobench::doNotOptimizeAway(output_buffer);
    });
}
