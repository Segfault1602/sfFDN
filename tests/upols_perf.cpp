#include "doctest.h"
#include "nanobench.h"

#include "upols.h"
#include <array>
#include <iostream>

#include <memory>

#include "filter.h"
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
    for (size_t j = 0; j < sos.size(); j++)
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
    std::cout << "Created test filter with " << sos.size() << " sections." << std::endl;

    return filter;
}
} // namespace

TEST_CASE("UPOLS")
{
    constexpr size_t kBlockSize = 64;

    constexpr size_t kFirLength = 2198;
    auto ref_filter = CreateTestFilter();
    std::vector<float> fir(kFirLength, 0.f);
    for (size_t i = 0; i < kFirLength; ++i)
    {
        // Fill the FIR filter with some test coefficients
        fir[i] = ref_filter->Tick(i == 0 ? 1.f : 0.f); // Use the filter to generate coefficients
    }

    sfFDN::UPOLS upols(kBlockSize, fir);

    upols.PrintPartition();

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kBlockSize, 0.f);

    nanobench::Bench bench;
    bench.title("UPOLS perf");
    bench.batch(kBlockSize);
    bench.minEpochIterations(200000);

    bench.run("UPOLS", [&] {
        // Process the block
        upols.Process(input, output);
        nanobench::doNotOptimizeAway(output);
    });
}
