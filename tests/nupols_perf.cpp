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
        auto b = std::span<const float>(&sos[j % sos.size()][0], 3);
        auto a = std::span<const float>(&sos[j % sos.size()][3], 3);
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
    constexpr uint32_t kBlockSize = 64;

    constexpr uint32_t kFirLength = 4096;
    auto ref_filter = CreateTestFilter();
    std::vector<float> fir(kFirLength, 0.f);
    for (auto i = 0u; i < kFirLength; ++i)
    {
        // Fill the FIR filter with some test coefficients
        fir[i] = ref_filter->Tick(i == 0 ? 1.f : 0.f); // Use the filter to generate coefficients
    }

    sfFDN::PartitionedConvolver nupols(kBlockSize, fir);

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kBlockSize, 0.f);

    nanobench::Bench bench;
    bench.title("PartitionedConvolver perf");
    bench.minEpochIterations(20000);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
    bench.run("PartitionedConvolver", [&] {
        // Process the block
        nupols.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}
