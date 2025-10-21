#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include "sffdn/sffdn.h"

#include "filter_coeffs.h"
#include "test_utils.h"

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
        std::span<const float> sos_span(sos.at(j % sos.size()));
        auto b = sos_span.first(3);
        auto a = sos_span.last(3);
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

    auto ref_filter = CreateTestFilter();
    auto fir = GetImpulseResponse(ref_filter.get());
    const uint32_t fir_length = fir.size();

    sfFDN::PartitionedConvolver partitioned_convolver(kBlockSize, fir);

    sfFDN::Fir fir_filter;
    fir_filter.SetCoefficients(fir);

    std::vector<float> input(fir_length + kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(fir_length + kBlockSize, 0.f);

    const uint32_t block_count = fir_length / kBlockSize;
    for (auto i = 0u; i < block_count; ++i)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, std::span(input).subspan(i * kBlockSize, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, std::span(output).subspan(i * kBlockSize, kBlockSize));
        // Process the block
        partitioned_convolver.Process(input_buffer, output_buffer);
    }

    std::vector<float> output_fir(fir_length + kBlockSize, 0.f);
    output_fir[0] = 1.f;
    auto fir_buffer = sfFDN::AudioBuffer(fir_length + kBlockSize, 1, output_fir);
    fir_filter.Process(fir_buffer, fir_buffer);

    float fir_energy = 0.f;
    float signal_error = 0.f;
    for (auto i = 0u; i < fir_length; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(fir[i], 1e-5));
        fir_energy += fir[i] * fir[i];
        signal_error += (output[i] - fir[i]) * (output[i] - fir[i]);

        REQUIRE_THAT(output_fir[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }

    float snr = 10.f * log10(fir_energy / signal_error);
    std::cout << "SNR: " << snr << " dB" << '\n';
}

TEST_CASE("PartitionedConvolver_Noise")
{
    constexpr uint32_t kBlockSize = 128;

    auto ref_filter = CreateTestFilter();
    auto fir = GetImpulseResponse(ref_filter.get());

    sfFDN::Fir fir_filter;
    fir_filter.SetCoefficients(fir);

    std::vector<float> input_chirp = ReadWavFile("./tests/data/chirp.wav");
    const uint32_t input_size = input_chirp.size();

    std::vector<float> filter_output(input_size, 0.f);
    sfFDN::AudioBuffer input_buffer(input_size, 1, input_chirp);
    sfFDN::AudioBuffer ref_output_buffer(input_size, 1, filter_output);

    fir_filter.Process(input_buffer, ref_output_buffer);

    sfFDN::PartitionedConvolver partitioned_convolver(kBlockSize, fir);

    std::vector<float> output(input_size, 0.f);

    const uint32_t block_count = input_size / kBlockSize;
    for (auto i = 0u; i < block_count; ++i)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, std::span(input_chirp).subspan(i * kBlockSize, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, std::span(output).subspan(i * kBlockSize, kBlockSize));
        // Process the block
        partitioned_convolver.Process(input_buffer, output_buffer);
    }

    float signal_energy = 0.f;
    float signal_error = 0.f;
    float max_error = 0.f;
    for (auto i = 0u; i < input_size - kBlockSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(filter_output[i], 1e-5));
        signal_energy += filter_output[i] * filter_output[i];
        signal_error += (output[i] - filter_output[i]) * (output[i] - filter_output[i]);
        max_error = std::max(max_error, std::abs(output[i] - filter_output[i]));
    }

    float snr = 10.f * log10(signal_energy / signal_error);
    std::cout << "PartitionedConvolver SNR: " << snr << " dB, Max Error: " << max_error << "\n";
}