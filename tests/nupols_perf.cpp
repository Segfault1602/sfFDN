#include "doctest.h"
#include "nanobench.h"

#include "nupols.h"
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

TEST_CASE("NUPOLS")
{
    constexpr size_t kBlockSize = 64;

    constexpr size_t kFirLength = 4096;
    auto ref_filter = CreateTestFilter();
    std::vector<float> fir(kFirLength, 0.f);
    for (size_t i = 0; i < kFirLength; ++i)
    {
        // Fill the FIR filter with some test coefficients
        fir[i] = ref_filter->Tick(i == 0 ? 1.f : 0.f); // Use the filter to generate coefficients
    }

    sfFDN::NUPOLS nupols(kBlockSize, fir, sfFDN::PartitionStrategy::kGardner);

    nupols.DumpInfo();

    std::vector<float> input(kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kBlockSize, 0.f);

    constexpr size_t kLoopCount = 50; // The NUPOLS algorithm doesn't do the same amount of work at each iteration, so
                                      // we run it multiple times to get a better average performance.

    nanobench::Bench bench;
    bench.title("NUPOLS perf");
    bench.batch(kBlockSize * kLoopCount);
    bench.minEpochIterations(4000);

    bench.run("NUPOLS", [&] {
        // Process the block
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        for (size_t i = 0; i < kLoopCount; ++i)
        {
            // Process the block
            nupols.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        }
    });
}
