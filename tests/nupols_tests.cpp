#include "doctest.h"

#include <array>
#include <iostream>

#include <memory>
#include <random>
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
    for (auto j = 0; j < sos.size(); j++)
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
    constexpr uint32_t kBlockSize = 128;

    auto ref_filter = CreateTestFilter();
    auto fir = GetImpulseResponse(ref_filter.get());
    const uint32_t kFirLength = fir.size();

    sfFDN::PartitionedConvolver PartitionedConvolver(kBlockSize, fir);

    InnerProdFIR inner_prod_fir(fir);

    std::vector<float> input(kFirLength + kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kFirLength + kBlockSize, 0.f);

    const uint32_t kBlockCount = kFirLength / kBlockSize;
    for (auto i = 0; i < kBlockCount; ++i)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input.data() + i * kBlockSize);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data() + i * kBlockSize);
        // Process the block
        PartitionedConvolver.Process(input_buffer, output_buffer);
    }

    std::vector<float> output_fir(kFirLength + kBlockSize, 0.f);
    output_fir[0] = 1.f;
    auto fir_buffer = sfFDN::AudioBuffer(kFirLength + kBlockSize, 1, output_fir.data());
    inner_prod_fir.Process(fir_buffer);

    float fir_energy = 0.f;
    float signal_error = 0.f;
    for (auto i = 0; i < kFirLength; ++i)
    {
        CHECK(output[i] == doctest::Approx(fir[i]));
        fir_energy += fir[i] * fir[i];
        signal_error += (output[i] - fir[i]) * (output[i] - fir[i]);

        CHECK(output_fir[i] == doctest::Approx(output[i]));
    }

    float snr = 10.f * log10(fir_energy / signal_error);
}

TEST_CASE("PartitionedConvolver_Noise")
{
    constexpr uint32_t kBlockSize = 128;

    auto ref_filter = CreateTestFilter();
    auto fir = GetImpulseResponse(ref_filter.get());
    const uint32_t kFirLength = fir.size();

    InnerProdFIR inner_prod_fir(fir);

    std::vector<float> input_chirp = ReadWavFile("./tests/data/chirp.wav");
    const uint32_t kInputSize = input_chirp.size();

    std::vector<float> filter_output(kInputSize, 0.f);
    sfFDN::AudioBuffer input_buffer(kInputSize, 1, input_chirp.data());
    sfFDN::AudioBuffer ref_output_buffer(kInputSize, 1, filter_output.data());

    std::copy(input_chirp.begin(), input_chirp.end(), filter_output.begin());
    inner_prod_fir.Process(ref_output_buffer);

    sfFDN::PartitionedConvolver PartitionedConvolver(kBlockSize, fir);

    std::vector<float> output(kInputSize, 0.f);

    const uint32_t kBlockCount = kInputSize / kBlockSize;
    for (auto i = 0; i < kBlockCount; ++i)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input_chirp.data() + i * kBlockSize);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data() + i * kBlockSize);
        // Process the block
        PartitionedConvolver.Process(input_buffer, output_buffer);
    }

    float signal_energy = 0.f;
    float signal_error = 0.f;
    float max_error = 0.f;
    for (auto i = 0; i < kInputSize - kBlockSize; ++i)
    {
        CHECK(output[i] == doctest::Approx(filter_output[i]));
        signal_energy += filter_output[i] * filter_output[i];
        signal_error += (output[i] - filter_output[i]) * (output[i] - filter_output[i]);
        max_error = std::max(max_error, std::abs(output[i] - filter_output[i]));
    }

    float snr = 10.f * log10(signal_energy / signal_error);
}