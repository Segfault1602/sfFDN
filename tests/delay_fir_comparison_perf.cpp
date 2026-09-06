#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "processor_perf_utils.h"
#include "rng.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("DelayFirComparisonPerf", "[delay][.diagnostic]")
{
    constexpr uint32_t kTapCount = 16U;
    constexpr uint32_t kMinTap = 0U;
    constexpr uint32_t kMaxTap = 8192U;
    constexpr uint32_t kBlockSize = 128U;

    std::vector<uint32_t> taps;
    taps.reserve(kTapCount);
    for (uint32_t tap = 0; tap < kTapCount; ++tap)
    {
        taps.push_back(kMinTap + ((tap * 509U) % (kMaxTap - kMinTap + 1U)));
    }

    std::vector<float> input(kBlockSize);
    std::vector<float> delay_output(kBlockSize);
    std::vector<float> fir_output(kBlockSize);
    sfFDN::test::perf::FillNoise(input);
    std::vector<float> coefficients(taps.size(), 1.F);

    sfFDN::Delay delay(0, kMaxTap + kBlockSize);
    REQUIRE(delay.AddNextInputs(input));
    delay.GetNextOutputsAt(taps, delay_output, coefficients);
    delay.AdvanceRead(kBlockSize);

    sfFDN::SparseFirOptions sparse_fir_options;
    for (size_t index = 0; index < taps.size(); ++index)
    {
        sparse_fir_options.coeffs.emplace_back(taps[index], coefficients[index]);
    }

    sfFDN::SparseFir fir(sparse_fir_options);
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(fir_output);
    fir.Process(input_buffer, output_buffer);

    for (uint32_t sample = 0; sample < kBlockSize; ++sample)
    {
        REQUIRE_THAT(delay_output[sample], Catch::Matchers::WithinAbs(fir_output[sample], 1e-5F));
    }

    nanobench::Bench bench;
    bench.title("Delay multitap vs SparseFir");
    bench.relative(true);
    bench.timeUnit(1us, "us");
    bench.minEpochTime(10ms);

    bench.run("Delay multitap", [&] {
        std::ranges::fill(delay_output, 0.F);
        delay.AddNextInputs(input);
        delay.GetNextOutputsAt(taps, delay_output, coefficients);
        delay.AdvanceRead(kBlockSize);
        nanobench::doNotOptimizeAway(delay_output);
    });

    bench.run("SparseFir", [&] {
        fir.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(fir_output);
    });
}
