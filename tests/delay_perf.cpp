#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "rng.h"
#include "sffdn/delay_interp.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("Delay")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kDelay = 2456;
    constexpr uint32_t kMaxDelay = 8192;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    sfFDN::Delay delay(kDelay, kMaxDelay);

    nanobench::Bench bench;
    bench.title("Delay Perf");
    bench.relative(true);
    bench.timeUnit(1us, "us");

    bench.minEpochIterations(500000);
    bench.run("Delay Tick", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay.Tick(input[i]);
        }
        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    bench.minEpochIterations(500000);
    bench.run("Delay block", [&] {
        delay.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(input_buffer);
        nanobench::doNotOptimizeAway(output_buffer);
    });

    sfFDN::DelayInterp delay_allpass(kDelay + 0.5f, kMaxDelay, sfFDN::DelayInterpolationType::Allpass);

    bench.minEpochIterations(10000);
    bench.run("DelayAllpass Tick", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay_allpass.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    bench.minEpochIterations(10000);
    bench.run("DelayAllpass block", [&] {
        delay_allpass.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(input_buffer);
        nanobench::doNotOptimizeAway(output_buffer);
    });

    sfFDN::DelayInterp delay_interp(kDelay + 0.5f, kMaxDelay, sfFDN::DelayInterpolationType::Linear);
    bench.minEpochIterations(10000);
    bench.run("DelayInterp Tick", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay_interp.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    bench.minEpochIterations(10000);
    bench.run("DelayInterp block", [&] {
        delay_interp.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(input_buffer);
        nanobench::doNotOptimizeAway(output_buffer);
    });

    sfFDN::DelayTimeVarying delay_time_varying(kDelay, kMaxDelay);
    delay_time_varying.SetMod(0.001f, 16.f);
    bench.minEpochIterations(10000);
    bench.run("DelayTimeVarying Tick", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay_time_varying.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    bench.minEpochIterations(10000);
    bench.run("DelayTimeVarying block", [&] {
        delay_time_varying.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(input_buffer);
        nanobench::doNotOptimizeAway(output_buffer);
    });
}

TEST_CASE("DelayBank")
{
    constexpr uint32_t kChannelCount = 16;
    constexpr std::array<uint32_t, kChannelCount> kDelays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593,
                                                             3253, 3343, 3547, 3559, 4483, 4507, 4663, 5483};

    constexpr uint32_t kBlockSize = 128;

    sfFDN::DelayBank delay_bank(kDelays, kBlockSize);

    std::vector<float> input(kBlockSize * kChannelCount, 0.f);
    std::vector<float> output(kBlockSize * kChannelCount, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    nanobench::Bench bench;
    bench.title("DelayBank Perf");
    // bench.batch(kBlockSize);
    bench.minEpochIterations(120000);
    bench.run("DelayBank", [&] {
        delay_bank.GetNextOutputs(output_buffer);
        delay_bank.AddNextInputs(input_buffer);
    });
}

TEST_CASE("DelayBank_BlockSize")
{
    constexpr uint32_t kChannelCount = 16;
    constexpr std::array<uint32_t, kChannelCount> kDelays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593,
                                                             3253, 3343, 3547, 3559, 4483, 4507, 4663, 5483};

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr uint32_t kInputSize = 1 << 12;

    std::vector<float> input(kInputSize * kChannelCount, 0.f);
    std::vector<float> output(kInputSize * kChannelCount, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    nanobench::Bench bench;
    bench.title("DelayBank Perf - Block Size");
    bench.batch(kInputSize);
    bench.relative(true);

    for (auto block_size : kBlockSizes)
    {
        bench.minEpochIterations(std::min(100 * block_size, 1000));
        sfFDN::DelayBank delay_bank(kDelays, block_size);

        bench.run("DelayBank BlockSize " + std::to_string(block_size), [&] {
            uint32_t block_count = kInputSize / block_size;
            for (auto j = 0u; j < block_count; ++j)
            {
                sfFDN::AudioBuffer input_buffer(
                    block_size, kChannelCount,
                    std::span<float>(input).subspan(j * block_size, block_size * kChannelCount));
                sfFDN::AudioBuffer output_buffer(
                    block_size, kChannelCount,
                    std::span<float>(output).subspan(j * block_size, block_size * kChannelCount));

                delay_bank.GetNextOutputs(output_buffer);
                delay_bank.AddNextInputs(input_buffer);
            }
        });
    }
}

TEST_CASE("Delay_MultiTap")
{
    constexpr uint32_t kTapCount = 16;
    constexpr uint32_t kMinTap = 0;
    constexpr uint32_t kMaxTap = 8192;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(kMinTap, kMaxTap);
    std::vector<uint32_t> taps;
    taps.reserve(kTapCount);
    for (auto i = 0u; i < kTapCount; ++i)
    {
        taps.push_back(dis(gen));
    }

    constexpr uint32_t kBlockSize = 128;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    std::vector<float> output_fir(kBlockSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    nanobench::Bench bench;
    bench.title("Delay MultiTap Perf");
    bench.relative(true);
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    std::vector<float> coeffs(taps.size(), 1.f);

    sfFDN::Delay delay_bank(0, kMaxTap + kBlockSize);
    delay_bank.AddNextInputs(input);
    delay_bank.GetNextOutputsAt(taps, output, coeffs);

    sfFDN::SparseFir fir;
    fir.SetCoefficients(coeffs, taps);
    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output_fir);
    fir.Process(input_buffer, output_buffer);

    // Sanity check
    for (auto i = 0u; i < output.size(); ++i)
    {
        auto out_delay = output[i];
        auto out_fir = output_fir[i];
        REQUIRE_THAT(out_delay, Catch::Matchers::WithinAbs(out_fir, 1e-5));
    }

    bench.run("Delay MultiTap", [&] {
        delay_bank.AddNextInputs(input);
        delay_bank.GetNextOutputsAt(taps, output, coeffs);
    });

    bench.run("FIR MultiTap", [&] { fir.Process(input_buffer, output_buffer); });
}
