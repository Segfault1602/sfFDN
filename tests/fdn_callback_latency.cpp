#include "production_workloads.h"

#include "rng.h"
#include "sffdn/audio_buffer.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <chrono>
#endif

namespace
{
uint64_t ReadTimestamp()
{
#if defined(__APPLE__)
    return mach_absolute_time();
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
#endif
}

double TicksToNanoseconds(uint64_t ticks)
{
#if defined(__APPLE__)
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t info{};
        mach_timebase_info(&info);
        return info;
    }();
    return static_cast<double>(ticks) * static_cast<double>(timebase.numer) / static_cast<double>(timebase.denom);
#else
    return static_cast<double>(ticks);
#endif
}

double Percentile(const std::vector<uint64_t>& sorted_durations, double percentile)
{
    const auto index =
        static_cast<size_t>(percentile * static_cast<double>(sorted_durations.size() - 1));
    return TicksToNanoseconds(sorted_durations[index]) / 1000.0;
}

struct Options
{
    uint32_t iterations;
    std::string workload_filter;
};

Options ParseOptions(std::span<char*> arguments)
{
    constexpr uint32_t kDefaultIterations = 20000;
    if (arguments.size() < 2)
    {
        return {.iterations = kDefaultIterations, .workload_filter = {}};
    }

    uint32_t iterations = 0;
    const std::string_view value(arguments[1]);
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), iterations);
    if (error != std::errc{} || end != value.data() + value.size() || iterations == 0)
    {
        std::cerr << "Usage: sfFDN.callback_latency [positive iteration count] [workload name substring]\n";
        return {};
    }
    return {.iterations = iterations, .workload_filter = arguments.size() >= 3 ? arguments[2] : ""};
}
} // namespace

int main(int argc, char** argv)
{
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#endif
    const std::span arguments(argv, static_cast<size_t>(argc));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    const Options options = ParseOptions(arguments);
    if (options.iterations == 0)
    {
        return 2;
    }

    constexpr uint32_t kWarmupIterations = 2000;
    std::printf("workload,iterations,callback_frames,p50_us,p95_us,p99_us,max_us,deadline_us,deadline_misses\n");

    for (auto& workload : CreateProductionFDNWorkloads())
    {
        if (!options.workload_filter.empty() && workload.name.find(options.workload_filter) == std::string::npos)
        {
            continue;
        }

        std::vector<float> input(workload.callback_size);
        std::vector<float> output(workload.callback_size);
        sfFDN::RNG generator;
        for (float& sample : input)
        {
            sample = generator();
        }

        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        for (auto iteration = 0u; iteration < kWarmupIterations; ++iteration)
        {
            workload.fdn->Process(input_buffer, output_buffer);
        }

        std::vector<uint64_t> durations(options.iterations);
        for (uint64_t& duration : durations)
        {
            const uint64_t start = ReadTimestamp();
            workload.fdn->Process(input_buffer, output_buffer);
            duration = ReadTimestamp() - start;
        }

        std::ranges::sort(durations);
        const double deadline_us =
            (static_cast<double>(workload.callback_size) / static_cast<double>(workload.sample_rate)) * 1'000'000.0;
        const uint64_t deadline_ticks = static_cast<uint64_t>(
            (deadline_us * 1000.0) / TicksToNanoseconds(1));
        const auto deadline_misses =
            std::ranges::count_if(durations, [deadline_ticks](uint64_t duration) { return duration > deadline_ticks; });

        std::printf("\"%s\",%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%lld\n", workload.name.c_str(), options.iterations,
                    workload.callback_size, Percentile(durations, 0.50), Percentile(durations, 0.95),
                    Percentile(durations, 0.99), TicksToNanoseconds(durations.back()) / 1000.0, deadline_us,
                    static_cast<long long>(deadline_misses));
    }
}
