#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kChannelCount = 8;
constexpr float kSampleRate = 96000.f;

std::vector<float> MakeNoise(uint32_t count)
{
    sfFDN::RNG generator;
    std::vector<float> noise(count, 0.f);
    for (float& sample : noise)
    {
        sample = generator();
    }
    return noise;
}
} // namespace

TEST_CASE("NonlinearPerf")
{
    auto input = MakeNoise(kBlockSize);
    std::vector<float> output(kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    sfFDN::ControllableFullWaveRectifier rectifier_plain(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = false, .dc_block = false, .sample_rate = kSampleRate});
    sfFDN::ControllableFullWaveRectifier rectifier_aa(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = false, .sample_rate = kSampleRate});
    sfFDN::ControllableFullWaveRectifier rectifier_full(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 1.f, .antialiasing = true, .dc_block = true, .sample_rate = kSampleRate});
    sfFDN::SignalDependentFractionalDelay sdfd(sfFDN::SignalDependentFractionalDelayOptions{.d = 1.f});
    sfFDN::RingModulator ring_mod(sfFDN::RingModulatorOptions{
        .frequency = 100.f / kSampleRate, .amplitude = std::numbers::sqrt2_v<float>, .initial_phase = 0.f});

    nanobench::Bench bench;
    bench.title("Shimmer nonlinearities, one channel");
    bench.timeUnit(1us, "us");
    // These kernels are a fraction of a microsecond per block, short enough that a run lands entirely inside one
    // CPU frequency state. With the defaults the reported err% is well under 1% while the same row moves by 6-7%
    // from one run to the next, which reads as a real change and is not one. The warmup absorbs the frequency ramp
    // and the longer epochs spread the samples across it.
    bench.warmup(2000);
    bench.minEpochTime(200ms);
    bench.relative(true);

    bench.run("CFWR", [&] { rectifier_plain.Process(input_buffer, output_buffer); });
    bench.run("CFWR + antialiasing", [&] { rectifier_aa.Process(input_buffer, output_buffer); });
    bench.run("CFWR + antialiasing + dc blocker", [&] { rectifier_full.Process(input_buffer, output_buffer); });
    bench.run("SDFD", [&] { sdfd.Process(input_buffer, output_buffer); });
    bench.run("RingModulator", [&] { ring_mod.Process(input_buffer, output_buffer); });
}

TEST_CASE("NonlinearBankPerf")
{
    auto input = MakeNoise(kChannelCount * kBlockSize);
    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    auto rectifier_bank = sfFDN::MakeMultichannelControllableFullWaveRectifier(
        sfFDN::MakeMultichannelControllableFullWaveRectifierOptions(1.f, kSampleRate, kChannelCount));
    auto sdfd_bank = sfFDN::MakeMultichannelSignalDependentFractionalDelay(
        sfFDN::MakeMultichannelSignalDependentFractionalDelayOptions(1.f, kChannelCount));
    auto ring_mod_bank = sfFDN::MakeMultichannelRingModulator(sfFDN::MakeMultichannelRingModulatorOptions(
        100.f / kSampleRate, std::numbers::sqrt2_v<float>, kChannelCount));

    // Half the channels active, so the cost of the pass-through slots shows up as well.
    auto half_bank = sfFDN::MakeMultichannelControllableFullWaveRectifier(
        sfFDN::MakeMultichannelControllableFullWaveRectifierOptions(1.f, kSampleRate, kChannelCount,
                                                                    kChannelCount / 2));

    nanobench::Bench bench;
    bench.title("Shimmer nonlinearity banks, eight channels");
    bench.timeUnit(1us, "us");
    // See NonlinearPerf: the defaults understate the run-to-run variance of these rows by a large factor.
    bench.warmup(2000);
    bench.minEpochTime(200ms);
    bench.relative(true);

    bench.run("CFWR bank", [&] { rectifier_bank->Process(input_buffer, output_buffer); });
    bench.run("CFWR bank, half the channels", [&] { half_bank->Process(input_buffer, output_buffer); });
    bench.run("SDFD bank", [&] { sdfd_bank->Process(input_buffer, output_buffer); });
    bench.run("RingModulator bank", [&] { ring_mod_bank->Process(input_buffer, output_buffer); });
}
