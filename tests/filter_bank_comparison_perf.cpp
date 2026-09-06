#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "filter_coeffs.h"
#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FilterBankComparisonPerf", "[filter][.diagnostic]")
{
    constexpr uint32_t kChannelCount = 16U;
    constexpr uint32_t kStageCount = 10U;
    constexpr uint32_t kBlockSize = 128U;
    const auto source = std::span(k_h001_AbsorbtionSOS[0]).first(kStageCount);

    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    std::vector<sfFDN::FilterCoefficients> bank_coefficients;
    bank_coefficients.reserve(kChannelCount * kStageCount);
    for (uint32_t channel = 0; channel < kChannelCount; ++channel)
    {
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(source);
        filter_bank->AddFilter(std::move(filter));
        bank_coefficients.insert(bank_coefficients.end(), source.begin(), source.end());
    }

    sfFDN::IIRFilterBank iir_filter_bank;
    iir_filter_bank.SetFilter(bank_coefficients, kChannelCount);

    std::vector<float> input(kChannelCount * kBlockSize);
    std::vector<float> output(input.size());
    sfFDN::test::perf::FillNoise(input);
    const sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    nanobench::Bench bench;
    bench.title("FilterBank implementation comparison N=16 stages=10 B=128");
    bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochTime(std::chrono::milliseconds(10));

    bench.run("FilterBank", [&] {
        filter_bank->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
    bench.run("IIRFilterBank", [&] {
        iir_filter_bank.Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });

#ifdef __APPLE__
    std::vector<double> vdsp_coefficients;
    vdsp_coefficients.reserve(kChannelCount * kStageCount * 5U);
    for (uint32_t channel = 0; channel < kChannelCount; ++channel)
    {
        for (const sfFDN::FilterCoefficients& coefficients : source)
        {
            const auto normalized = coefficients.Normalize();
            vdsp_coefficients.push_back(normalized.b0);
            vdsp_coefficients.push_back(normalized.b1);
            vdsp_coefficients.push_back(normalized.b2);
            vdsp_coefficients.push_back(normalized.a1);
            vdsp_coefficients.push_back(normalized.a2);
        }
    }
    vDSP_biquadm_Setup setup = vDSP_biquadm_CreateSetup(vdsp_coefficients.data(), kStageCount, kChannelCount);
    REQUIRE(setup != nullptr);
    std::array<const float*, kChannelCount> input_pointers{};
    std::array<float*, kChannelCount> output_pointers{};
    for (uint32_t channel = 0; channel < kChannelCount; ++channel)
    {
        input_pointers[channel] = input_buffer.GetChannelSpan(channel).data();
        output_pointers[channel] = output_buffer.GetChannelSpan(channel).data();
    }
    bench.run("vDSP_biquadm", [&] {
        vDSP_biquadm(setup, input_pointers.data(), 1, output_pointers.data(), 1, kBlockSize);
        nanobench::doNotOptimizeAway(output);
    });
    vDSP_biquadm_DestroySetup(setup);
#endif
}
