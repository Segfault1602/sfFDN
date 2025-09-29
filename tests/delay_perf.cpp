#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "rng.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("Delay")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kDelay = 4663;
    constexpr uint32_t kMaxDelay = 8192;

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    sfFDN::Delay delay(kDelay, kMaxDelay);

    nanobench::Bench bench;
    bench.title("Delay Perf");
    bench.relative(true);
    bench.timeUnit(1us, "us");

    bench.minEpochIterations(50000);
    bench.run("Delay Linear", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay.Tick(input[i]);
        }
        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    bench.minEpochIterations(500000);
    bench.run("Delay block", [&] {
        delay.GetNextOutputs(output);
        delay.AddNextInputs(input);
        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::DelayAllpass delay_allpass(kDelay + 0.5f, kMaxDelay);

    bench.minEpochIterations(10000);
    bench.run("DelayAllpass Linear", [&] {
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            output[i] = delay_allpass.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(input);
        nanobench::doNotOptimizeAway(output);
    });

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
    bench.minEpochIterations(10000);
    bench.run("DelayAllpass block", [&] {
        delay_allpass.Process(input_buffer, output_buffer);
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
    constexpr uint32_t kDelayCount = kDelays.size();

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
    constexpr uint32_t kDelayCount = kDelays.size();
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
    bench.minEpochIterations(120);
    bench.relative(true);

    for (auto block_size : kBlockSizes)
    {
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
