#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <span>

#include <sndfile.h>

#include "filter_coeffs.h"
#include "sffdn/sffdn.h"

#include "test_utils.h"

namespace
{

std::unique_ptr<sfFDN::FDN> CreateReferenceFDN(bool transpose)
{
    constexpr uint32_t block_size = 256;
    constexpr uint32_t N = 6;
    constexpr std::array<float, N> input_gains = {0.072116069, 0.24890353,   0.97228086,
                                                  -0.38236806, -0.057921566, -0.39115807};
    constexpr std::array<float, N> output_gains = {-0.46316639, -0.36613876, 0.30902779,
                                                   0.30143532,  -0.49200505, 0.58704174};
    constexpr std::array<uint32_t, N> delays = {593, 743, 929, 1153, 1399, 1699};

    constexpr std::array<float, N * N> mixing_matrix = {
        0.590748429298401,  0.457586556673050,  0.0557801127433777, -0.148047655820847,  -0.478258520364761,
        -0.433439940214157, -0.158531382679939, 0.433001756668091,  -0.0591235160827637, 0.626041889190674,
        0.430089294910431,  -0.454946815967560, -0.665803074836731, 0.195845842361450,   0.568070054054260,
        -0.251500934362412, -0.263658404350281, -0.250756144523621, 0.239477828145027,   -0.236257210373878,
        0.618841290473938,  0.622415661811829,  -0.255638062953949, 0.226088821887970,   0.266185045242310,
        -0.500568747520447, 0.346136510372162,  -0.255272954702377, 0.454669415950775,   -0.535609304904938,
        0.233208581805229,  0.508312821388245,  0.409773439168930,  -0.265208065509796,  0.494672924280167,
        0.451974451541901};

    auto fdn = std::make_unique<sfFDN::FDN>(N, block_size, transpose);
    fdn->SetInputGains(input_gains);
    fdn->SetOutputGains(output_gains);
    fdn->SetDirectGain(0.f);
    fdn->SetDelays(delays);

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(N);
    mix_mat->SetMatrix(mixing_matrix);

    fdn->SetFeedbackMatrix(std::move(mix_mat));

    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0; i < N; i++)
    {
        auto sos = k_h001_AbsorbtionSOS[i];
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();

        std::vector<float> coeffs;
        for (auto j = 0; j < sos.size(); j++)
        {
            auto b = std::span<const float>(&sos[j][0], 3);
            auto a = std::span<const float>(&sos[j][3], 3);
            coeffs.push_back(b[0] / a[0]);
            coeffs.push_back(b[1] / a[0]);
            coeffs.push_back(b[2] / a[0]);
            coeffs.push_back(a[1] / a[0]);
            coeffs.push_back(a[2] / a[0]);
        }

        filter->SetCoefficients(sos.size(), coeffs);

        filter_bank->AddFilter(std::move(filter));
    }

    fdn->SetFilterBank(std::move(filter_bank));

    std::vector<float> coeffs;
    for (auto i = 0; i < k_h001_EqualizationSOS.size(); i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(k_h001_EqualizationSOS.size(), coeffs);
    fdn->SetTCFilter(std::move(filter));
    return fdn;
}

} // namespace

TEST_CASE("FDN")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t ITER = SR;

    auto fdn = CreateReferenceFDN(false);

    // Send some garbage data first to test that `Clear()` works as expected
    std::vector<float> garbage(4096, 1.f);
    sfFDN::AudioBuffer garbage_buffer(4096, 1, garbage);
    fdn->Process(garbage_buffer, garbage_buffer);

    fdn->Clear();

    std::vector<float> input(ITER, 0.f);
    input[0] = 1.f;
    std::vector<float> output(ITER, 0.f);
    auto clone_input = input;
    auto clone_output = output;

    sfFDN::AudioBuffer input_buffer(ITER, 1, input.data());
    sfFDN::AudioBuffer output_buffer(ITER, 1, output.data());
    fdn->Process(input_buffer, output_buffer);

    sfFDN::AudioBuffer clone_input_buffer(ITER, 1, clone_input.data());
    sfFDN::AudioBuffer clone_output_buffer(ITER, 1, clone_output.data());
    auto clone_fdn = fdn->Clone();
    clone_fdn->Process(clone_input_buffer, clone_output_buffer);

    {
        constexpr const char* expected_output_filename = "./tests/data/fdn_gold_test.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t REQUIRE_limit = std::min(output.size(), expected_output.size());

        for (auto i = 0; i < REQUIRE_limit; ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-4));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);

            // Check that the cloned FDN is also doing the right thing
            REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(output[i], 1e-7));
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        std::cout << "FDN SNR: " << snr << " dB\n";
    }
}

TEST_CASE("FDN_Transposed")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t ITER = SR;

    auto fdn = CreateReferenceFDN(true);

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);
    input[0] = 1.f;

    sfFDN::AudioBuffer input_buffer(ITER, 1, input.data());
    sfFDN::AudioBuffer output_buffer(ITER, 1, output.data());
    fdn->Process(input_buffer, output_buffer);

    {
        constexpr const char* expected_output_filename = "./tests/data/fdn_gold_test_transposed.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t REQUIRE_limit = std::min(output.size(), expected_output.size());

        for (auto i = 0; i < REQUIRE_limit; ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-4));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        std::cout << "FDN_transpose SNR: " << snr << " dB\n";
    }
}

TEST_CASE("FDN_FIR")
{
    constexpr uint32_t SR = 48000;
    constexpr uint32_t block_size = 64;
    constexpr uint32_t N = 6;
    constexpr uint32_t ITER = ((SR / block_size) + 1) * block_size;
    constexpr std::array<uint32_t, N> delays = {593, 743, 929, 1153, 1399, 1699};

    auto fdn = CreateReferenceFDN(false);

    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto delay : delays)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_" + std::to_string(delay) + ".wav");
        auto convolver = std::make_unique<sfFDN::PartitionedConvolver>(block_size, fir);

        filter_bank->AddFilter(std::move(convolver));
    }

    fdn->SetFilterBank(std::move(filter_bank));

    {
        auto eq_fir = ReadWavFile("./tests/data/equalization_fir.wav");
        auto tc_filter = std::make_unique<sfFDN::PartitionedConvolver>(block_size, eq_fir);
        fdn->SetTCFilter(std::move(tc_filter));
    }

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);

    input[0] = 1.f;

    for (auto i = 0; i < input.size(); i += block_size)
    {
        sfFDN::AudioBuffer input_buffer(block_size, 1, input.data() + i);
        sfFDN::AudioBuffer output_buffer(block_size, 1, output.data() + i);

        fdn->Process(input_buffer, output_buffer);
    }

    WriteWavFile("fdn_fir_test.wav", output);

    {
        constexpr const char* expected_output_filename = "./tests/data/fdn_gold_fir_test.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t REQUIRE_limit = std::min(output.size(), expected_output.size());

        for (auto i = 0; i < REQUIRE_limit; ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 5e-4));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        SUCCEED("FDN (FIR) SNR: " << snr << " dB");
    }
}

TEST_CASE("FDN_Chirp")
{
    constexpr uint32_t SR = 48000;

    auto fdn = CreateReferenceFDN(false);

    std::vector<float> input = ReadWavFile("./tests/data/chirp_ramp.wav");

    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(input.size(), 1, input.data());
    sfFDN::AudioBuffer output_buffer(output.size(), 1, output.data());
    fdn->Process(input_buffer, output_buffer);

    WriteWavFile("fdn_chirp_test.wav", output);

    {
        constexpr const char* expected_output_filename = "./tests/data/chirp_reverb.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t REQUIRE_limit = std::min(output.size(), expected_output.size());

        for (auto i = 0; i < REQUIRE_limit; ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-2));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        std::cout << "FDN (chirp) SNR: " << snr << " dB\n";
    }
}
