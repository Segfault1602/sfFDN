#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "production_workloads.h"
#include "rng.h"
#include "sffdn/audio_buffer.h"

#include <chrono>
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
