#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>
#include <memory>

#include "sffdn/sffdn.h"

#include "filter_coeffs.h"

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
std::unique_ptr<sfFDN::CascadedBiquads> CreateTestFilter()
{
    // Create a simple filter for testing purposes
    auto filter = std::make_unique<sfFDN::CascadedBiquads>();
    std::vector<float> coeffs;
    auto sos = k_h001_AbsorbtionSOS[0];
    for (auto j = 0u; j < sos.size(); j++)
    {
        auto stage_span = std::span(sos[j % sos.size()]);
        auto b = stage_span.first(3);
        auto a = stage_span.last(3);
        coeffs.push_back(b[0] / a[0]);
        coeffs.push_back(b[1] / a[0]);
        coeffs.push_back(b[2] / a[0]);
        coeffs.push_back(a[1] / a[0]);
        coeffs.push_back(a[2] / a[0]);
    }

    filter->SetCoefficients(sos.size(), coeffs);

    return filter;
}
} // namespace

TEST_CASE("PartitionedConvolver")
{
    constexpr uint32_t kBlockSize = 128;

    constexpr uint32_t kFirLength = 24000;
    auto ref_filter = CreateTestFilter();
    std::vector<float> fir(kFirLength, 0.f);
    for (auto i = 0u; i < kFirLength; ++i)
    {
        // Fill the FIR filter with some test coefficients
        fir[i] = ref_filter->Tick(i == 0 ? 1.f : 0.f); // Use the filter to generate coefficients
    }

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kBlockSize, 0.f);

    nanobench::Bench bench;
    bench.title("PartitionedConvolver perf");
    bench.minEpochIterations(20000);
    bench.timeUnit(1us, "us");
    bench.relative(true);

    for (uint32_t rep_count = 2; rep_count <= 32; rep_count *= 2)
    {
        sfFDN::PartitionedConvolver nupols(kBlockSize, fir, rep_count);
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        bench.run(nupols.GetShortInfo(), [&] {
            // Process the block
            nupols.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }

    // Check for max time
    constexpr auto kRepCount = 8u;
    sfFDN::PartitionedConvolver nupols(kBlockSize, fir, kRepCount);
    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    std::vector<float> durations;
    durations.reserve(1000);
    for (auto i = 0u; i < 1000; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        nupols.Process(input_buffer, output_buffer);
        auto end = std::chrono::steady_clock::now();
        double duration_us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count();
        durations.push_back(duration_us);
    }

    constexpr double kMaxAllowedDurationUs = 1.0e6 / (48000.0 / kBlockSize);
    for (const auto& duration : durations)
    {
        REQUIRE(duration < kMaxAllowedDurationUs);
        // std::cout << duration << "\n";
    }
}
