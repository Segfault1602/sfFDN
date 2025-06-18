#include "doctest.h"

#include "nupols.h"
#include <array>
#include <iostream>

#include <memory>
#include <random>
#include <span>
#include <vector>

#include "audio_buffer.h"
#include "filter.h"
#include "filter_coeffs.h"
#include "filter_utils.h"
#include "test_utils.h"

namespace
{
std::unique_ptr<fdn::CascadedBiquads> CreateTestFilter()
{
    // Create a simple filter for testing purposes
    auto filter = std::make_unique<fdn::CascadedBiquads>();
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

    return filter;
}
} // namespace

TEST_CASE("NUPOLS")
{
    constexpr size_t kBlockSize = 128;

    auto ref_filter = CreateTestFilter();
    auto fir = fdn::GetImpulseResponse(ref_filter.get());
    const size_t kFirLength = fir.size();

    fdn::NUPOLS nupols(kBlockSize, fir, fdn::PartitionStrategy::kGardner);

    nupols.DumpInfo();

    InnerProdFIR inner_prod_fir(fir);

    std::vector<float> input(kFirLength + kBlockSize, 0.f);
    input[0] = 1.f;
    std::vector<float> output(kFirLength + kBlockSize, 0.f);

    const size_t kBlockCount = kFirLength / kBlockSize;
    for (size_t i = 0; i < kBlockCount; ++i)
    {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input.data() + i * kBlockSize);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output.data() + i * kBlockSize);
        // Process the block
        nupols.Process(input_buffer, output_buffer);
    }

    std::vector<float> output_fir(kFirLength + kBlockSize, 0.f);
    output_fir[0] = 1.f;
    auto fir_buffer = fdn::AudioBuffer(kFirLength + kBlockSize, 1, output_fir.data());
    inner_prod_fir.Process(fir_buffer);

    float fir_energy = 0.f;
    float signal_error = 0.f;
    for (size_t i = 0; i < kFirLength; ++i)
    {
        CHECK(output[i] == doctest::Approx(fir[i]));
        fir_energy += fir[i] * fir[i];
        signal_error += (output[i] - fir[i]) * (output[i] - fir[i]);

        CHECK(output_fir[i] == doctest::Approx(output[i]));
    }

    float snr = 10.f * log10(fir_energy / signal_error);
    std::cout << "NUPOLS SNR: " << snr << " dB" << std::endl;
}

TEST_CASE("NUPOLS_Noise")
{
    constexpr size_t kBlockSize = 128;

    auto ref_filter = CreateTestFilter();
    auto fir = fdn::GetImpulseResponse(ref_filter.get());
    const size_t kFirLength = fir.size();

    InnerProdFIR inner_prod_fir(fir);

    std::vector<float> input_chirp = ReadWavFile("./tests/chirp.wav");
    const size_t kInputSize = input_chirp.size();

    std::vector<float> filter_output(kInputSize, 0.f);
    fdn::AudioBuffer input_buffer(kInputSize, 1, input_chirp.data());
    fdn::AudioBuffer ref_output_buffer(kInputSize, 1, filter_output.data());

    std::copy(input_chirp.begin(), input_chirp.end(), filter_output.begin());
    inner_prod_fir.Process(ref_output_buffer);

    fdn::NUPOLS nupols(kBlockSize, fir, fdn::PartitionStrategy::kGardner);

    nupols.DumpInfo();

    std::vector<float> output(kInputSize, 0.f);

    const size_t kBlockCount = kInputSize / kBlockSize;
    for (size_t i = 0; i < kBlockCount; ++i)
    {
        fdn::AudioBuffer input_buffer(kBlockSize, 1, input_chirp.data() + i * kBlockSize);
        fdn::AudioBuffer output_buffer(kBlockSize, 1, output.data() + i * kBlockSize);
        // Process the block
        nupols.Process(input_buffer, output_buffer);
    }

    float signal_energy = 0.f;
    float signal_error = 0.f;
    float max_error = 0.f;
    for (size_t i = 0; i < kInputSize - kBlockSize; ++i)
    {
        CHECK(output[i] == doctest::Approx(filter_output[i]));
        signal_energy += filter_output[i] * filter_output[i];
        signal_error += (output[i] - filter_output[i]) * (output[i] - filter_output[i]);
        max_error = std::max(max_error, std::abs(output[i] - filter_output[i]));
    }

    float snr = 10.f * log10(signal_energy / signal_error);
    std::cout << "NUPOLS SNR: " << snr << " dB" << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
}