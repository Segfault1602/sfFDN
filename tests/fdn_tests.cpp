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
    constexpr uint32_t kBlockSize = 256;
    constexpr uint32_t kFDNOrder = 6;
    constexpr std::array<float, kFDNOrder> kInputGains = {0.072116069f, 0.24890353f,   0.97228086f,
                                                          -0.38236806f, -0.057921566f, -0.39115807f};
    constexpr std::array<float, kFDNOrder> kOutputGains = {-0.46316639f, -0.36613876f, 0.30902779f,
                                                           0.30143532f,  -0.49200505f, 0.58704174f};
    constexpr std::array<float, kFDNOrder> kDelays = {593, 743, 929, 1153, 1399, 1699};

    const std::vector<float> kMixingMatrix = {
        0.590748429298401f,  0.457586556673050f,  0.0557801127433777f, -0.148047655820847f,  -0.478258520364761f,
        -0.433439940214157f, -0.158531382679939f, 0.433001756668091f,  -0.0591235160827637f, 0.626041889190674f,
        0.430089294910431f,  -0.454946815967560f, -0.665803074836731f, 0.195845842361450f,   0.568070054054260f,
        -0.251500934362412f, -0.263658404350281f, -0.250756144523621f, 0.239477828145027f,   -0.236257210373878f,
        0.618841290473938f,  0.622415661811829f,  -0.255638062953949f, 0.226088821887970f,   0.266185045242310f,
        -0.500568747520447f, 0.346136510372162f,  -0.255272954702377f, 0.454669415950775f,   -0.535609304904938f,
        0.233208581805229f,  0.508312821388245f,  0.409773439168930f,  -0.265208065509796f,  0.494672924280167f,
        0.451974451541901f};

    auto fdn = std::make_unique<sfFDN::FDN>(kFDNOrder, kBlockSize, transpose);
    fdn->SetInputGains(kInputGains);
    fdn->SetOutputGains(kOutputGains);
    fdn->SetDirectGain(0.f);
    fdn->SetDelays(kDelays);

    sfFDN::ScalarFeedbackMatrixOptions mix_mat_config;
    mix_mat_config.matrix_size = kFDNOrder;
    mix_mat_config.custom_matrix = kMixingMatrix;

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(mix_mat_config);

    fdn->SetFeedbackMatrix(std::move(mix_mat));

    auto filter_bank = std::make_unique<sfFDN::IIRFilterBank>();
    std::vector<sfFDN::FilterCoefficients> iir_coeffs;
    for (auto i = 0u; i < kFDNOrder; i++)
    {
        auto sos = k_h001_AbsorbtionSOS.at(i);

        for (auto& stage : sos)
        {
            iir_coeffs.push_back(stage);
        }
    }

    filter_bank->SetFilter(iir_coeffs, kFDNOrder);

    fdn->SetLoopFilter(std::move(filter_bank));

    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(k_h001_EqualizationSOS);
    fdn->SetTCFilter(std::move(filter));
    return fdn;
}

} // namespace

TEST_CASE("FDN")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kIter = kSampleRate;

    auto fdn = CreateReferenceFDN(false);

    // Send some garbage data first to test that `Clear()` works as expected
    // std::vector<float> garbage(4096, 1.f);
    // sfFDN::AudioBuffer garbage_buffer(4096, 1, garbage);
    // fdn->Process(garbage_buffer, garbage_buffer);

    // fdn->Clear();

    std::vector<float> input(kIter, 0.f);
    input[0] = 1.f;

    std::vector<float> output(kIter, 0.f);
    auto clone_input = input;
    auto clone_output = output;

    sfFDN::AudioBuffer input_buffer(kIter, 1, input);
    sfFDN::AudioBuffer output_buffer(kIter, 1, output);
    fdn->Process(input_buffer, output_buffer);

    sfFDN::AudioBuffer clone_input_buffer(kIter, 1, clone_input);
    sfFDN::AudioBuffer clone_output_buffer(kIter, 1, clone_output);
    auto clone_fdn = fdn->Clone();
    clone_fdn->Process(clone_input_buffer, clone_output_buffer);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/fdn_gold_test.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(kExpectedOutputFilename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == kSampleRate);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t testing_boundary = std::min(output.size(), expected_output.size());

        for (auto i = 0u; i < testing_boundary; ++i)
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
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kIter = kSampleRate;

    auto fdn = CreateReferenceFDN(true);

    std::vector<float> input(kIter, 0.f);
    std::vector<float> output(kIter, 0.f);
    input[0] = 1.f;

    sfFDN::AudioBuffer input_buffer(kIter, 1, input);
    sfFDN::AudioBuffer output_buffer(kIter, 1, output);
    fdn->Process(input_buffer, output_buffer);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/fdn_gold_test_transposed.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(kExpectedOutputFilename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == kSampleRate);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t testing_boundary = std::min(output.size(), expected_output.size());

        for (auto i = 0u; i < testing_boundary; ++i)
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
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 64;
    constexpr uint32_t kN = 6;
    constexpr uint32_t kIter = ((kSampleRate / kBlockSize) + 1) * kBlockSize;
    constexpr std::array<uint32_t, kN> kDelays = {593, 743, 929, 1153, 1399, 1699};

    auto fdn = CreateReferenceFDN(false);

    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto delay : kDelays)
    {
        auto fir = ReadWavFile("./tests/data/att_fir_" + std::to_string(delay) + ".wav");
        auto convolver = std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, fir);

        filter_bank->AddFilter(std::move(convolver));
    }

    fdn->SetLoopFilter(std::move(filter_bank));

    {
        auto eq_fir = ReadWavFile("./tests/data/equalization_fir.wav");
        auto tc_filter = std::make_unique<sfFDN::PartitionedConvolver>(kBlockSize, eq_fir);
        fdn->SetTCFilter(std::move(tc_filter));
    }

    std::vector<float> input(kIter, 0.f);
    std::vector<float> output(kIter, 0.f);

    input[0] = 1.f;

    for (auto i = 0u; i < input.size(); i += kBlockSize)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, std::span(input).subspan(i, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, std::span(output).subspan(i, kBlockSize));

        fdn->Process(input_buffer, output_buffer);
    }

    WriteWavFile("fdn_fir_test.wav", output);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/fdn_gold_fir_test.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(kExpectedOutputFilename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == kSampleRate);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        size_t test_boundary = std::min(output.size(), expected_output.size());

        for (auto i = 0u; i < test_boundary; ++i)
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
    constexpr uint32_t kSampleRate = 48000;

    auto fdn = CreateReferenceFDN(false);

    std::vector<float> input = ReadWavFile("./tests/data/chirp_ramp.wav");

    std::vector<float> output(input.size(), 0.f);
    sfFDN::AudioBuffer input_buffer(input.size(), 1, input);
    sfFDN::AudioBuffer output_buffer(output.size(), 1, output);
    fdn->Process(input_buffer, output_buffer);

    WriteWavFile("fdn_chirp_test.wav", output);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/chirp_reverb.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(kExpectedOutputFilename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == kSampleRate);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);

        float signal_energy = 0.f;
        float signal_error = 0.f;

        uint32_t test_boundary = std::min(output.size(), expected_output.size());

        for (auto i = 0u; i < test_boundary; ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-2));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        std::cout << "FDN (chirp) SNR: " << snr << " dB\n";
    }
}

TEST_CASE("FDNConfig_Example")
{
    sfFDN::FDNConfig config;
    config.fdn_size = 8;
    config.direct_gain = 1.f;
    config.block_size = 128;
    config.sample_rate = 48000;

    sfFDN::DelayBankOptions delay_bank_options{
        .delays = sfFDN::GetDelayLengths(config.fdn_size, 500, 3000, sfFDN::DelayLengthType::Random),
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None};

    config.delay_bank_config = delay_bank_options;

    sfFDN::ParallelGainsOptions input_gains_options{.mode = sfFDN::ParallelGainsMode::Split,
                                                    .gains = std::vector<float>(config.fdn_size, 0.5f),
                                                    .time_varying_config = {}};

    config.input_block_config.parallel_gains_config = input_gains_options;

    sfFDN::ScalarFeedbackMatrixOptions feedback_matrix_options{.matrix_size = config.fdn_size,
                                                               .type = sfFDN::ScalarMatrixType::Hadamard};

    config.feedback_matrix_config = feedback_matrix_options;

    sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_options;
    sfFDN::HomogenousFilterOptions homogenous_filter_options{
        .t60 = 1.f, .delay = 0.f, .sample_rate = config.sample_rate};

    // If only 1 filter is found in AttenuationFilterBankOptions, CreateFDNFromConfig() will reuse the same filter for
    // all channels, updating the delay value based on the corresponding delay line length for each channel.
    attenuation_filter_bank_options.filter_configs.emplace_back(homogenous_filter_options);

    config.loop_filter_configs.emplace_back(attenuation_filter_bank_options);

    sfFDN::ParallelGainsOptions output_gains_options{.mode = sfFDN::ParallelGainsMode::Merge,
                                                     .gains = std::vector<float>(config.fdn_size, 0.5f),
                                                     .time_varying_config = {}};

    config.output_block_config.parallel_gains_config = output_gains_options;

    auto fdn = sfFDN::CreateFDNFromConfig(config);

    std::vector<float> input(48000, 0.f);
    input[0] = 1.f;

    std::vector<float> output(48000, 0.f);

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    fdn->Process(input_buffer, output_buffer);

    WriteWavFile("fdn_config_example.wav", output);

    nlohmann::json json_config = config;

    sfFDN::FDNConfig deserialized_config = json_config.get<sfFDN::FDNConfig>();
    auto deserialized_fdn = sfFDN::CreateFDNFromConfig(deserialized_config);

    std::vector<float> deserialized_output(48000, 0.f);
    sfFDN::AudioBuffer deserialized_output_buffer(deserialized_output);
    deserialized_fdn->Process(input_buffer, deserialized_output_buffer);

    for (size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE_THAT(deserialized_output[i], Catch::Matchers::WithinAbs(output[i], 1e-6));
    }
}