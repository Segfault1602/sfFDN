#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include "sffdn/oscillator.h"

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
void StdSin(std::span<float> output, float phase_increment)
{
    float phase = 0;
    for (float& sample : output)
    {
        sample = std::sinf(phase * 2.0f * std::numbers::pi);
        phase += phase_increment;
    }
}

} // namespace

TEST_CASE("SineWave performance", "[oscillator]")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;
    constexpr float kFrequency = 10.f / static_cast<float>(kSampleRate);

    sfFDN::SineWave generate(kFrequency);
    sfFDN::SineWave multiply(kFrequency);
    sfFDN::SineWave multiply_accumulate(kFrequency);

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "SineWave", 100ms, 2000);
    sfFDN::test::perf::SetChannelSampleBatch(bench, kBlockSize);
    bench.relative(true);

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize, 0.f);
    sfFDN::test::perf::FillNoise(input);

    bench.run("Generate", [&]() {
        generate.Generate(output);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("Multiply", [&]() {
        multiply.Multiply(input, output);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("MultiplyAccumulate", [&]() {
        multiply_accumulate.MultiplyAccumulate(input, output);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("std::sinf", [&]() {
        StdSin(output, kFrequency);
        nanobench::doNotOptimizeAway(output);
    });
}
