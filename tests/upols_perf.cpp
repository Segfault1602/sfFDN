#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>

#include <memory>

#include "sffdn/sffdn.h"
#include "upols.h"

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

TEST_CASE("UPOLS")
{
    constexpr uint32_t kBlockSize = 64;

    constexpr uint32_t kFirLength = 2198;
    auto ref_filter = CreateTestFilter();
    std::vector<float> fir(kFirLength, 0.f);
    for (auto i = 0u; i < kFirLength; ++i)
    {
        // Fill the FIR filter with some test coefficients
        fir[i] = ref_filter->Tick(i == 0 ? 1.f : 0.f); // Use the filter to generate coefficients
    }

    sfFDN::UPOLS upols(kBlockSize, fir);

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kBlockSize, 0.f);

    nanobench::Bench bench;
    bench.title("Uniform Partition Convolution - Perf");
    // bench.batch(kBlockSize);
    bench.minEpochIterations(20000);

    bench.run("UPOLS", [&] {
        // Process the block
        upols.Process(input, output);
        nanobench::doNotOptimizeAway(output);
    });
}
