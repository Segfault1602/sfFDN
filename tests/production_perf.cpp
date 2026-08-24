#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "production_workloads.h"
#include "rng.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/partitioned_convolver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
void RunProductionBenchmark(nanobench::Bench& bench, const std::string& name, sfFDN::FDN& fdn,
                            uint32_t callback_size)
{
    std::vector<float> input(callback_size);
    std::vector<float> output(callback_size);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    bench.run(name, [&] {
        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        fdn.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

std::vector<float> CreateProductionRIR(uint32_t sample_count)
{
    std::vector<float> rir(sample_count);
    sfFDN::RNG generator;
    for (auto sample = 0u; sample < sample_count; ++sample)
    {
        const float decay = std::exp(-6.f * static_cast<float>(sample) / static_cast<float>(sample_count));
        rir[sample] = generator() * decay;
    }
    rir[0] = 1.f;
    return rir;
}
} // namespace

TEST_CASE("ProductionFDNPerf", "[production-perf]")
{
    nanobench::Bench bench;
    bench.title("Production FDN workloads");
    bench.timeUnit(1us, "us");
    bench.warmup(100);
    bench.minEpochTime(50ms);

    for (auto& workload : CreateProductionFDNWorkloads())
    {
        RunProductionBenchmark(bench, workload.name, *workload.fdn, workload.callback_size);
    }
}

TEST_CASE("ProductionConvolverPerf", "[production-perf]")
{
    constexpr uint32_t kBlockSize = 1024;
    constexpr uint32_t kLoopCount = 512;
    constexpr std::array<uint32_t, 3> kRirLengths = {48000, 96000, 240000};

    nanobench::Bench bench;
    bench.title("Production partitioned convolution workloads");
    bench.timeUnit(1us, "us");
    bench.warmup(16);
    bench.minEpochTime(50ms);
    bench.batch(kLoopCount);

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    for (const uint32_t rir_length : kRirLengths)
    {
        auto rir = CreateProductionRIR(rir_length);
        sfFDN::PartitionedConvolver convolver(kBlockSize, rir);
        bench.run("RIR=" + std::to_string(rir_length) + " block=1024 " + convolver.GetShortInfo(), [&] {
            for (auto iteration = 0u; iteration < kLoopCount; ++iteration)
            {
                convolver.Process(input_buffer, output_buffer);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("ProductionConvolverPartitionPerf", "[production-perf]")
{
    constexpr uint32_t kBlockSize = 1024;
    constexpr uint32_t kLoopCount = 512;
    constexpr std::array<uint32_t, 3> kRirLengths = {48000, 96000, 240000};
    constexpr std::array<uint32_t, 5> kRepCounts = {2, 4, 8, 16, 32};

    nanobench::Bench bench;
    bench.title("Production partition schedule comparison");
    bench.timeUnit(1us, "us");
    bench.warmup(16);
    bench.minEpochTime(50ms);
    bench.batch(kLoopCount);

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    for (const uint32_t rir_length : kRirLengths)
    {
        const auto rir = CreateProductionRIR(rir_length);
        for (const uint32_t rep_count : kRepCounts)
        {
            sfFDN::PartitionedConvolver convolver(kBlockSize, rir, rep_count);
            bench.run("RIR=" + std::to_string(rir_length) + " rep=" + std::to_string(rep_count) + " " +
                          convolver.GetShortInfo(),
                      [&] {
                          for (auto iteration = 0u; iteration < kLoopCount; ++iteration)
                          {
                              convolver.Process(input_buffer, output_buffer);
                          }
                          nanobench::doNotOptimizeAway(output);
                      });
        }
    }
}

TEST_CASE("ProductionConvolverLatency", "[production-perf]")
{
    constexpr uint32_t kBlockSize = 1024;
    constexpr uint32_t kIterationCount = 8192;
    constexpr std::array<uint32_t, 3> kRirLengths = {48000, 96000, 240000};
    constexpr std::array<uint32_t, 2> kRepCounts = {8, 16};

    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    std::printf("rir_samples,rep_count,p50_us,p95_us,p99_us,max_us\n");
    for (const uint32_t rir_length : kRirLengths)
    {
        const auto rir = CreateProductionRIR(rir_length);
        for (const uint32_t rep_count : kRepCounts)
        {
            sfFDN::PartitionedConvolver convolver(kBlockSize, rir, rep_count);
            for (auto warmup = 0u; warmup < 512; ++warmup)
            {
                convolver.Process(input_buffer, output_buffer);
            }

            std::vector<double> durations(kIterationCount);
            for (double& duration : durations)
            {
                const auto start = std::chrono::steady_clock::now();
                convolver.Process(input_buffer, output_buffer);
                const auto end = std::chrono::steady_clock::now();
                duration = std::chrono::duration<double, std::micro>(end - start).count();
            }
            std::ranges::sort(durations);

            const auto percentile = [&durations](double value) {
                return durations[static_cast<size_t>(value * static_cast<double>(durations.size() - 1))];
            };
            std::printf("%u,%u,%.3f,%.3f,%.3f,%.3f\n", rir_length, rep_count, percentile(0.50),
                        percentile(0.95), percentile(0.99), durations.back());
        }
    }
}
