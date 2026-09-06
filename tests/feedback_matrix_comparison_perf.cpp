#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
std::vector<sfFDN::ModulationOptions> MakeModulationConfig(uint32_t order)
{
    std::vector<sfFDN::ModulationOptions> config(order / 2U);
    for (uint32_t rotation = 0; rotation < config.size(); ++rotation)
    {
        config[rotation] = {
            .frequency = 1.F / static_cast<float>(sfFDN::kDefaultSampleRate),
            .amplitude = 0.7F,
            .initial_phase = static_cast<float>((rotation * 7U) % order) / static_cast<float>(order),
        };
    }
    return config;
}

void RunFDN(sfFDN::FDN& fdn, std::string_view name, nanobench::Bench& bench)
{
    constexpr uint32_t kBlockSize = 128U;
    std::vector<float> input(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    bench.run(std::string(name), [&] {
        std::ranges::fill(output, 0.F);
        fdn.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
} // namespace

TEST_CASE("FeedbackMatrixComparisonPerf", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 16U;
    constexpr uint32_t kBlockSize = 128U;

    std::vector<float> input(static_cast<size_t>(kOrder) * kBlockSize);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kOrder, output);

    sfFDN::ScalarFeedbackMatrix hadamard({.matrix_size = kOrder, .type = sfFDN::ScalarMatrixType::Hadamard});
    sfFDN::TimeVaryingFeedbackMatrix time_varying_hadamard({
        .matrix_size = kOrder,
        .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
        .time_varying_config = MakeModulationConfig(kOrder),
    });
    sfFDN::TimeVaryingFeedbackMatrix time_varying_schur({
        .matrix_size = kOrder,
        .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config = MakeModulationConfig(kOrder),
        .rng_seed = 4242U,
    });

    nanobench::Bench bench;
    bench.title("Feedback matrix comparison N=16 B=128");
    bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochTime(10ms);

    bench.run("Scalar Hadamard", [&] {
        hadamard.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
    bench.run("TimeVarying Hadamard", [&] {
        time_varying_hadamard.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
    bench.run("TimeVarying RealSchur", [&] {
        time_varying_schur.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("FeedbackMatrixFDNComparisonPerf", "[fdn]")
{
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kOrder = 16U;

    auto static_fdn = CreateFDN(kBlockSize, kOrder);
    static_fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(
        sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = kOrder, .type = sfFDN::ScalarMatrixType::Hadamard}));

    auto time_varying_fdn = CreateFDN(kBlockSize, kOrder);
    time_varying_fdn->SetFeedbackMatrix(
        std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(sfFDN::TimeVaryingFeedbackMatrixOptions{
            .matrix_size = kOrder,
            .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
            .time_varying_config = MakeModulationConfig(kOrder),
        }));

    auto time_varying_fdn_schur = CreateFDN(kBlockSize, kOrder);
    time_varying_fdn_schur->SetFeedbackMatrix(
        std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(sfFDN::TimeVaryingFeedbackMatrixOptions{
            .matrix_size = kOrder,
            .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
            .time_varying_config = MakeModulationConfig(kOrder),
            .rng_seed = 4242U,
        }));

    nanobench::Bench bench;
    bench.title("FDN feedback matrix comparison N=16 B=128");
    bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochTime(10ms);

    RunFDN(*static_fdn, "FDN Scalar Hadamard", bench);
    RunFDN(*time_varying_fdn, "FDN TimeVarying Hadamard", bench);
    RunFDN(*time_varying_fdn_schur, "FDN TimeVarying RealSchur", bench);
}
