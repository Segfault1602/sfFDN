#pragma once

#include "nanobench.h"
#include "rng.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sfFDN::test::perf
{

inline constexpr std::array<uint32_t, 5> kChannelCounts = {4U, 8U, 16U, 32U, 64U};

inline bool EnvironmentFlagEnabled(const char* name)
{
    // Benchmark configuration is read during single-threaded test setup, before any timed work.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* value = std::getenv(name);
    return value != nullptr && !std::string_view(value).empty() && std::string_view(value) != "0";
}

inline std::span<const uint32_t> BlockSizes()
{
    static constexpr std::array<uint32_t, 1> kDefaultBlockSizes = {128U};
    static constexpr std::array<uint32_t, 4> kExtendedBlockSizes = {32U, 64U, 128U, 256U};
    return EnvironmentFlagEnabled("SFFDN_PERF_BLOCK_SWEEP") ? std::span<const uint32_t>(kExtendedBlockSizes)
                                                            : std::span<const uint32_t>(kDefaultBlockSizes);
}

inline bool SmokeModeEnabled()
{
    return EnvironmentFlagEnabled("SFFDN_PERF_SMOKE");
}

inline bool ComplexityEnforcementEnabled()
{
    return EnvironmentFlagEnabled("SFFDN_PERF_ENFORCE_COMPLEXITY");
}

inline void ConfigureThroughputBench(ankerl::nanobench::Bench& bench, std::string_view title,
                                     std::chrono::nanoseconds min_epoch_time = std::chrono::milliseconds(10),
                                     uint64_t warmup_iterations = 100)
{
    bench.title(std::string(title));
    bench.timeUnit(std::chrono::nanoseconds(1), "ns");
    bench.unit("channel-sample");

    if (SmokeModeEnabled())
    {
        bench.epochs(1);
        bench.warmup(1);
        bench.minEpochTime(std::chrono::microseconds(100));
        return;
    }

    bench.epochs(7);
    bench.warmup(warmup_iterations);
    bench.minEpochTime(min_epoch_time);
}

inline void ConfigureComplexityBench(ankerl::nanobench::Bench& bench, std::string_view title,
                                     std::chrono::nanoseconds min_epoch_time = std::chrono::milliseconds(10),
                                     uint64_t warmup_iterations = 100)
{
    bench.title(std::string(title));
    bench.timeUnit(std::chrono::microseconds(1), "us");

    if (SmokeModeEnabled())
    {
        bench.epochs(1);
        bench.warmup(1);
        bench.minEpochTime(std::chrono::microseconds(100));
        return;
    }

    bench.epochs(7);
    bench.warmup(warmup_iterations);
    bench.minEpochTime(min_epoch_time);
}

inline void SetChannelSampleBatch(ankerl::nanobench::Bench& bench, uint32_t block_size, uint32_t channel_count = 1U,
                                  uint32_t block_count = 1U)
{
    bench.batch(static_cast<uint64_t>(block_size) * channel_count * block_count);
}

inline void FillNoise(std::span<float> data, uint64_t seed = 0x9E3779B9U)
{
    RNG generator(seed);
    for (float& sample : data)
    {
        sample = generator();
    }
}

inline std::string FormatComplexityFits(const std::vector<ankerl::nanobench::BigO>& fits)
{
    std::ostringstream stream;
    stream << fits;
    return stream.str();
}

} // namespace sfFDN::test::perf
