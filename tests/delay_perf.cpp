#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "sffdn/audio_buffer.h"
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
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = dist(generator);
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
    constexpr uint32_t N = 16;
    constexpr std::array<uint32_t, N> kDelays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593,
                                                 3253, 3343, 3547, 3559, 4483, 4507, 4663, 5483};

    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kDelayCount = kDelays.size();

    sfFDN::DelayBank delay_bank(kDelays, kBlockSize);

    std::vector<float> input(kBlockSize * N, 0.f);
    std::vector<float> output(kBlockSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

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
    constexpr uint32_t N = 16;
    constexpr std::array<uint32_t, N> kDelays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593,
                                                 3253, 3343, 3547, 3559, 4483, 4507, 4663, 5483};

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr uint32_t kDelayCount = kDelays.size();
    constexpr uint32_t kInputSize = 1 << 12;

    std::vector<float> input(kInputSize * N, 0.f);
    std::vector<float> output(kInputSize * N, 0.f);
    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    nanobench::Bench bench;
    bench.title("DelayBank Perf - Block Size");
    bench.batch(kInputSize);
    bench.minEpochIterations(120);
    bench.relative(true);

    for (auto i = 0u; i < kBlockSizes.size(); ++i)
    {
        uint32_t kBlockSize = kBlockSizes[i];
        sfFDN::DelayBank delay_bank(kDelays, kBlockSize);

        bench.run("DelayBank BlockSize " + std::to_string(kBlockSize), [&] {
            uint32_t block_count = kInputSize / kBlockSize;
            for (auto j = 0u; j < block_count; ++j)
            {
                sfFDN::AudioBuffer input_buffer(kBlockSize, 1,
                                                std::span<float>(input).subspan(j * kBlockSize, kBlockSize));
                sfFDN::AudioBuffer output_buffer(kBlockSize, 1,
                                                 std::span<float>(output).subspan(j * kBlockSize, kBlockSize));

                delay_bank.GetNextOutputs(output_buffer);
                delay_bank.AddNextInputs(input_buffer);
            }
        });
    }
}
