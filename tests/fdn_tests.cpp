#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <sndfile.h>

#include "filter_coeffs.h"
#include "sffdn/sffdn.h"
#include <sffdn/serialization.h>

#include "allocation_counter.h"
#include "test_utils.h"

namespace
{

std::unique_ptr<sfFDN::FDN> CreatePyFDNGoldFDN()
{
    constexpr uint32_t kBlockSize = 4;
    constexpr uint32_t kFDNOrder = 4;
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    constexpr std::array<float, kFDNOrder> kInputGains = {0.6f, -0.4f, 0.8f, -0.7f};
    constexpr std::array<float, kFDNOrder> kOutputGains = {0.5f, -0.6f, 0.7f, -0.3f};
    constexpr std::array<float, kFDNOrder> kDelays = {7.f, 11.f, 13.f, 17.f};
    // pyFDN supplies this deliberately non-symmetric matrix in row-major order.
    constexpr std::array<float, kFDNOrder * kFDNOrder> kMixingMatrix = {kInvSqrt2, 0.f,        0.5f,  0.5f,  //
                                                                        0.f,       -kInvSqrt2, 0.5f,  -0.5f, //
                                                                        kInvSqrt2, 0.f,        -0.5f, -0.5f, //
                                                                        0.f,       -kInvSqrt2, -0.5f, 0.5f};
    constexpr std::array<sfFDN::FilterCoefficients, kFDNOrder> kLoopFilters = {
        sfFDN::FilterCoefficients{0.3f, 0.f, 0.f, 1.f, -0.7f, 0.f},
        sfFDN::FilterCoefficients{0.4f, 0.f, 0.f, 1.f, -0.6f, 0.f},
        sfFDN::FilterCoefficients{0.5f, 0.f, 0.f, 1.f, -0.5f, 0.f},
        sfFDN::FilterCoefficients{0.6f, 0.f, 0.f, 1.f, -0.4f, 0.f}};
    constexpr std::array<sfFDN::FilterCoefficients, 1> kToneFilter = {
        sfFDN::FilterCoefficients{0.2f, 0.1f, 0.f, 1.f, -1.f, 0.34f}};

    auto fdn = std::make_unique<sfFDN::FDN>(kFDNOrder, kBlockSize, false);
    fdn->SetInputGains(kInputGains);
    fdn->SetOutputGains(kOutputGains);
    fdn->SetDirectGain(0.5f);
    fdn->SetDelays(kDelays);

    sfFDN::ScalarFeedbackMatrixOptions mix_mat_config{
        .source = sfFDN::MatrixData{kFDNOrder, std::vector<float>(kMixingMatrix.begin(), kMixingMatrix.end())}};
    fdn->SetFeedbackMatrix(std::make_unique<sfFDN::ScalarFeedbackMatrix>(mix_mat_config));

    auto filter_bank = std::make_unique<sfFDN::IIRFilterBank>();
    filter_bank->SetFilter(kLoopFilters, kFDNOrder);
    fdn->SetLoopFilter(std::move(filter_bank));

    auto tone_filter = std::make_unique<sfFDN::CascadedBiquads>();
    tone_filter->SetCoefficients(kToneFilter);
    fdn->SetTCFilter(std::move(tone_filter));
    return fdn;
}

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

    sfFDN::ScalarFeedbackMatrixOptions mix_mat_config{.source = sfFDN::MatrixData{kFDNOrder, kMixingMatrix}};

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

sfFDN::FDNConfig MakeFactoryConfig(bool transposed = false)
{
    constexpr uint32_t kOrder = 4;
    sfFDN::FDNConfig config{};
    config.fdn_size = kOrder;
    config.transposed = transposed;
    config.direct_gain = 0.F;
    config.block_size = 8;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {8.F, 9.F, 10.F, 11.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config =
        sfFDN::ScalarFeedbackMatrixOptions{.source = sfFDN::GeneratedMatrixOptions{
                                               .matrix_size = kOrder,
                                               .generator = sfFDN::ScalarMatrixType::Hadamard,
                                           }};
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    return config;
}

sfFDN::FDNConfig MakeOneSampleCharacterizationConfig()
{
    constexpr uint32_t kOrder = 4;
    sfFDN::FDNConfig config{};
    config.fdn_size = kOrder;
    config.direct_gain = 0.F;
    config.block_size = 1U;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {1.F, 1.F, 1.F, 1.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .gains = {1.F, 0.F, 0.F, 0.F},
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = kOrder,
                .generator = sfFDN::ScalarMatrixType::Identity,
            },
    };
    config.output_block_config.parallel_gains_config = {
        .gains = {1.F, 0.F, 0.F, 0.F},
        .time_varying_config = {},
    };
    return config;
}

sfFDN::AttenuationFilterBankOptions MakeAttenuationBank(size_t count)
{
    sfFDN::AttenuationFilterBankOptions bank;
    for (size_t index = 0; index < count; ++index)
    {
        bank.filter_configs.emplace_back(
            sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 8.F, .sample_rate = 48000.F});
    }
    return bank;
}

std::vector<float> RenderFactoryConfig(const sfFDN::FDNConfig& config)
{
    constexpr uint32_t kBlockCount = 4;
    std::vector<float> input(config.block_size * kBlockCount, 0.F);
    std::vector<float> output(input.size(), 0.F);
    input[0] = 1.F;
    auto fdn = sfFDN::CreateFDNFromConfig(config);

    for (uint32_t block = 0; block < kBlockCount; ++block)
    {
        const uint32_t offset = block * config.block_size;
        sfFDN::AudioBuffer input_buffer(config.block_size, 1U, std::span(input).subspan(offset, config.block_size));
        sfFDN::AudioBuffer output_buffer(config.block_size, 1U, std::span(output).subspan(offset, config.block_size));
        std::ranges::fill(output_buffer.GetChannelSpan(0), 0.F);
        fdn->Process(input_buffer, output_buffer);
    }
    return output;
}

} // namespace

TEST_CASE("FDN matches the pyFDN golden reference", "[fdn]")
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kIter = 4096;

    auto fdn = CreatePyFDNGoldFDN();
    auto clone_fdn = fdn->Clone();

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
    clone_fdn->Process(clone_input_buffer, clone_output_buffer);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/fdn_gold_test.wav";
        // libsndfile requires a zeroed SF_INFO: in SFM_READ mode a nonzero format field makes sf_open fail.
        SF_INFO sfinfo{};
        SNDFILE* expected_output_file = sf_open(kExpectedOutputFilename, SFM_READ, &sfinfo);

        REQUIRE(expected_output_file != nullptr);

        REQUIRE(sfinfo.channels == 1);
        REQUIRE(sfinfo.samplerate == kSampleRate);
        REQUIRE(sfinfo.frames == static_cast<sf_count_t>(kIter));
        REQUIRE((sfinfo.format & SF_FORMAT_SUBMASK) == SF_FORMAT_FLOAT);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        REQUIRE(read == sfinfo.frames);
        sf_close(expected_output_file);
        REQUIRE(output.size() == expected_output.size());

        float signal_energy = 0.f;
        float signal_error = 0.f;

        for (auto i = 0u; i < output.size(); ++i)
        {
            REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-5));
            signal_energy += expected_output[i] * expected_output[i];
            signal_error += (output[i] - expected_output[i]) * (output[i] - expected_output[i]);

            // Check that the cloned FDN is also doing the right thing
            REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(output[i], 1e-7));
        }
        float snr = 10.f * std::log10(signal_energy / signal_error);
        INFO("FDN SNR: " << snr << " dB");
    }
}

TEST_CASE("FDN transposed topology matches its golden reference", "[fdn]")
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
        // libsndfile requires a zeroed SF_INFO: in SFM_READ mode a nonzero format field makes sf_open fail.
        SF_INFO sfinfo{};
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
        INFO("FDN_transpose SNR: " << snr << " dB");
    }
}

TEST_CASE("FDN with FIR filters matches its golden reference", "[fdn]")
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

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/fdn_gold_fir_test.wav";
        // libsndfile requires a zeroed SF_INFO: in SFM_READ mode a nonzero format field makes sf_open fail.
        SF_INFO sfinfo{};
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

TEST_CASE("FDN reproduces the chirp golden file", "[fdn]")
{
    constexpr uint32_t kSampleRate = 48000;

    auto fdn = CreateReferenceFDN(false);

    std::vector<float> input = ReadWavFile("./tests/data/chirp_ramp.wav");

    std::vector<float> output(input.size(), 0.f);
    sfFDN::AudioBuffer input_buffer(input.size(), 1, input);
    sfFDN::AudioBuffer output_buffer(output.size(), 1, output);
    fdn->Process(input_buffer, output_buffer);

    {
        constexpr const char* kExpectedOutputFilename = "./tests/data/chirp_reverb.wav";
        // libsndfile requires a zeroed SF_INFO: in SFM_READ mode a nonzero format field makes sf_open fail.
        SF_INFO sfinfo{};
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
        INFO("FDN (chirp) SNR: " << snr << " dB");
    }
}

TEST_CASE("FDNConfig round-trips a rendered network", "[fdn]")
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

    sfFDN::StageGainsOptions input_gains_options{.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                 .time_varying_config = {}};

    config.input_block_config.parallel_gains_config = input_gains_options;

    sfFDN::ScalarFeedbackMatrixOptions feedback_matrix_options{.source = sfFDN::GeneratedMatrixOptions{
                                                                   .matrix_size = config.fdn_size,
                                                                   .generator = sfFDN::ScalarMatrixType::Hadamard,
                                                               }};

    config.feedback_matrix_config = feedback_matrix_options;

    sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_options;
    sfFDN::HomogenousFilterOptions homogenous_filter_options{
        .t60 = 1.f, .delay = 0.f, .sample_rate = config.sample_rate};

    // If only 1 filter is found in AttenuationFilterBankOptions, CreateFDNFromConfig() will reuse the same filter for
    // all channels, updating the delay value based on the corresponding delay line length for each channel.
    attenuation_filter_bank_options.filter_configs.emplace_back(homogenous_filter_options);

    config.loop_filter_configs.emplace_back(attenuation_filter_bank_options);

    sfFDN::StageGainsOptions output_gains_options{.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                  .time_varying_config = {}};

    config.output_block_config.parallel_gains_config = output_gains_options;

    auto fdn = sfFDN::CreateFDNFromConfig(config);

    std::vector<float> input(48000, 0.f);
    input[0] = 1.f;

    std::vector<float> output(48000, 0.f);

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    fdn->Process(input_buffer, output_buffer);

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
TEST_CASE("FDNConfig validates and round-trips multichannel Dattorro delay networks", "[fdn]")
{
    constexpr uint32_t kFdnSize = 8;
    constexpr float kSampleRate = 48000.f;

    sfFDN::FDNConfig config;
    config.fdn_size = kFdnSize;
    config.direct_gain = 0.f;
    config.block_size = 128;
    config.sample_rate = kSampleRate;

    config.delay_bank_config = sfFDN::DelayBankOptions{
        .delays = sfFDN::GetDelayLengths(config.fdn_size, 500, 3000, sfFDN::DelayLengthType::Random),
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None};

    config.input_block_config.parallel_gains_config = {.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                       .time_varying_config = {}};

    config.feedback_matrix_config =
        sfFDN::ScalarFeedbackMatrixOptions{.source = sfFDN::GeneratedMatrixOptions{
                                               .matrix_size = config.fdn_size,
                                               .generator = sfFDN::ScalarMatrixType::Hadamard,
                                           }};

    sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_options;
    attenuation_filter_bank_options.filter_configs.emplace_back(
        sfFDN::HomogenousFilterOptions{.t60 = 1.f, .delay = 0.f, .sample_rate = config.sample_rate});
    config.loop_filter_configs.emplace_back(attenuation_filter_bank_options);

    // A decorrelated vibrato per channel, sitting in the feedback loop after the static delay bank. Vibrato is the
    // only modulated preset with a gain of exactly 1 at every frequency: it has no feedback and no blend, so it is a
    // pure modulated delay. The presets that carry feedback (WhiteChorus, Flanger) reach roughly +15 dB once
    // modulated and make the network diverge; see MakeMultichannelDattorroDelayOptions() and the
    // "DattorroDelay preset gain in a feedback loop" test case.
    config.loop_filter_configs.emplace_back(
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::Vibrato, kSampleRate, kFdnSize));

    config.output_block_config.parallel_gains_config = {.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                        .time_varying_config = {}};

    auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn != nullptr);

    std::vector<float> input(static_cast<size_t>(kSampleRate), 0.f);
    input[0] = 1.f;
    std::vector<float> output(input.size(), 0.f);

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    fdn->Process(input_buffer, output_buffer);

    // The bank adds a modulated delay to every branch of the loop, so the first thing to check is that the whole
    // thing is still stable.
    float peak = 0.f;
    for (const float sample : output)
    {
        REQUIRE(std::isfinite(sample));
        peak = std::max(peak, std::abs(sample));
    }
    REQUIRE(peak > 0.f);

    // The tail must still be decaying a second in, rather than sustaining or growing.
    const auto energy = [&output](size_t begin, size_t end) {
        double sum = 0.0;
        for (size_t i = begin; i < end; ++i)
        {
            sum += static_cast<double>(output[i]) * output[i];
        }
        return sum;
    };
    const double early = energy(4800, 9600);
    const double late = energy(38400, 43200);
    REQUIRE(late < early);

    // The whole config, including the Dattorro bank, must survive a JSON round trip and rebuild identically.
    nlohmann::json json_config = config;
    auto deserialized_fdn = sfFDN::CreateFDNFromConfig(json_config.get<sfFDN::FDNConfig>());

    std::vector<float> deserialized_output(output.size(), 0.f);
    sfFDN::AudioBuffer deserialized_output_buffer(deserialized_output);
    deserialized_fdn->Process(input_buffer, deserialized_output_buffer);

    for (size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE_THAT(deserialized_output[i], Catch::Matchers::WithinAbs(output[i], 1e-6));
    }

    // A bank whose channel count does not match the FDN size must be rejected.
    sfFDN::FDNConfig bad_config = config;
    bad_config.loop_filter_configs[1] =
        sfFDN::MakeMultichannelDattorroDelayOptions(sfFDN::DattorroEffectType::WhiteChorus, kSampleRate, kFdnSize - 1);
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(bad_config), std::runtime_error);
}

TEST_CASE("FDNConfig validates time-varying Schroeder allpass networks", "[fdn]")
{
    constexpr uint32_t kFdnSize = 4;
    constexpr uint32_t kSampleCount = 240000;
    constexpr uint32_t kWindowSize = 20000;

    const auto make_config = [](bool attenuated) {
        sfFDN::FDNConfig config;
        config.fdn_size = kFdnSize;
        config.transposed = false;
        config.direct_gain = 0.F;
        config.block_size = 128;
        config.sample_rate = 48000.F;
        config.delay_bank_config = {
            .delays = {149.F, 211.F, 263.F, 293.F},
            .block_size = config.block_size,
            .interpolation_type = sfFDN::DelayInterpolationType::None,
        };
        config.input_block_config.parallel_gains_config = {
            .gains = std::vector<float>(kFdnSize, 0.5F),
            .time_varying_config = {},
        };
        config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
            .source = sfFDN::GeneratedMatrixOptions{
                .matrix_size = kFdnSize,
                .generator = sfFDN::ScalarMatrixType::Hadamard,
            }};
        config.output_block_config.parallel_gains_config = {
            .gains = std::vector<float>(kFdnSize, 0.5F),
            .time_varying_config = {},
        };

        sfFDN::MultichannelProcessorOptions bank;
        for (uint32_t channel = 0; channel < kFdnSize; ++channel)
        {
            bank.channels.emplace_back(sfFDN::TimeVaryingSchroederAllpassSectionOptions{
                .delays = {5.F + (2.F * static_cast<float>(channel))},
                .gains = {0.45F - (0.05F * static_cast<float>(channel))},
                .time_varying_config = {{.frequency =
                                             (0.5F + (0.125F * static_cast<float>(channel))) / config.sample_rate,
                                         .amplitude = 0.3F,
                                         .initial_phase = static_cast<float>(channel) / static_cast<float>(kFdnSize)}},
            });
        }
        config.loop_filter_configs.emplace_back(bank);

        if (attenuated)
        {
            sfFDN::AttenuationFilterBankOptions attenuation;
            attenuation.filter_configs.emplace_back(
                sfFDN::HomogenousFilterOptions{.t60 = 1.5F, .delay = 0.F, .sample_rate = config.sample_rate});
            config.attenuation_filter_bank_config = attenuation;
        }
        return config;
    };

    const auto render = [](const sfFDN::FDNConfig& config) {
        auto fdn = sfFDN::CreateFDNFromConfig(config);
        std::vector<float> input(kSampleCount, 0.F);
        std::vector<float> output(kSampleCount, 0.F);
        input[0] = 1.F;
        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(output);
        fdn->Process(input_buffer, output_buffer);
        return output;
    };

    const auto rms = [](std::span<const float> signal, size_t start) {
        double energy = 0.0;
        for (const float sample : signal.subspan(start, kWindowSize))
        {
            energy += static_cast<double>(sample) * sample;
        }
        return std::sqrt(energy / static_cast<double>(kWindowSize));
    };

    const auto lossless = render(make_config(false));
    REQUIRE(std::ranges::all_of(lossless, [](float sample) { return std::isfinite(sample); }));
    float peak = 0.F;
    for (const float sample : lossless)
    {
        peak = std::max(peak, std::abs(sample));
    }
    REQUIRE(peak < 2.F);
    const double lossless_ratio = rms(lossless, kSampleCount - kWindowSize) / rms(lossless, 80000);
    INFO("Lossless late/early RMS ratio: " << lossless_ratio);
    REQUIRE(lossless_ratio > 0.9);
    REQUIRE(lossless_ratio < 1.1);

    const auto attenuated = render(make_config(true));
    REQUIRE(std::ranges::all_of(attenuated, [](float sample) { return std::isfinite(sample); }));
    REQUIRE(rms(attenuated, kSampleCount - kWindowSize) < rms(attenuated, 20000));

    auto bad_config = make_config(false);
    auto& bad_bank = std::get<sfFDN::MultichannelProcessorOptions>(bad_config.loop_filter_configs[0]);
    bad_bank.channels.pop_back();
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(bad_config), std::runtime_error);

    bad_config = make_config(false);
    auto& invalid_bank = std::get<sfFDN::MultichannelProcessorOptions>(bad_config.loop_filter_configs[0]);
    auto& invalid_section =
        std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(invalid_bank.channels[0].value());
    invalid_section.gains[0] = 0.8F;
    invalid_section.time_varying_config[0].amplitude = 0.2F;
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(bad_config), std::runtime_error);
}

TEST_CASE("FDN supports arbitrary block lengths and duplicates its mono output", "[fdn]")
{
    constexpr uint32_t kSampleCount = 13;
    auto whole_fdn = CreatePyFDNGoldFDN();
    auto chunked_fdn = CreatePyFDNGoldFDN();
    std::array<float, kSampleCount> input{};
    input[0] = 1.f;
    std::array<float, kSampleCount> whole_output{};
    std::array<float, kSampleCount * 3> multichannel_output{};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer whole_output_buffer(whole_output);
    sfFDN::AudioBuffer multichannel_output_buffer(kSampleCount, 3, multichannel_output);
    whole_fdn->Process(input_buffer, whole_output_buffer);
    for (uint32_t offset = 0; offset < kSampleCount; offset += 5)
    {
        const uint32_t count = std::min(5u, kSampleCount - offset);
        const sfFDN::AudioBuffer input_block = input_buffer.Offset(offset, count);
        sfFDN::AudioBuffer output_block = multichannel_output_buffer.Offset(offset, count);
        chunked_fdn->Process(input_block, output_block);
    }

    for (uint32_t channel = 0; channel < 3; ++channel)
    {
        const auto channel_output = multichannel_output_buffer.GetChannelSpan(channel);
        for (size_t i = 0; i < whole_output.size(); ++i)
        {
            REQUIRE(channel_output[i] == Catch::Approx(whole_output[i]));
        }
    }
}

TEST_CASE("FDN output composition depends on the configured output path", "[fdn]")
{
    std::array<float, 1> impulse = {1.F};
    std::array<float, 1> silence = {0.F};
    const sfFDN::AudioBuffer impulse_buffer(impulse);
    const sfFDN::AudioBuffer silence_buffer(silence);

    const auto render_second_sample = [&](sfFDN::FDNConfig config) {
        auto fdn = sfFDN::CreateFDNFromConfig(config);
        std::array<float, 1> first_output{};
        sfFDN::AudioBuffer first_output_buffer(first_output);
        fdn->Process(impulse_buffer, first_output_buffer);

        std::array<float, 1> dirty_output = {10.F};
        sfFDN::AudioBuffer dirty_output_buffer(dirty_output);
        fdn->Process(silence_buffer, dirty_output_buffer);
        return dirty_output[0];
    };

    SECTION("bare merge accumulates wet output into the destination")
    {
        REQUIRE(render_second_sample(MakeOneSampleCharacterizationConfig()) == Catch::Approx(11.F));
    }

    SECTION("post-output processor overwrites the destination")
    {
        auto config = MakeOneSampleCharacterizationConfig();
        config.output_block_config.single_channel_processors.emplace_back(sfFDN::FirOptions{.coeffs = {2.F}});
        REQUIRE(render_second_sample(config) == Catch::Approx(2.F));
    }

    SECTION("tone correction processes pre-existing destination contents")
    {
        auto config = MakeOneSampleCharacterizationConfig();
        config.tone_correction_filters.emplace_back(sfFDN::FirOptions{.coeffs = {2.F}});
        REQUIRE(render_second_sample(config) == Catch::Approx(22.F));
    }
}

TEST_CASE("FDN Clear restores a fresh configured network and Clone is cleared", "[fdn]")
{
    constexpr uint32_t kSampleCount = 32;
    auto fdn = CreatePyFDNGoldFDN();
    auto fresh = CreatePyFDNGoldFDN();
    std::array<float, kSampleCount> impulse{};
    impulse[0] = 1.f;
    std::array<float, kSampleCount> warm_output{};
    std::array<float, kSampleCount> cleared_output{};
    std::array<float, kSampleCount> fresh_output{};
    sfFDN::AudioBuffer const input_buffer(impulse);
    sfFDN::AudioBuffer warm_output_buffer(warm_output);
    sfFDN::AudioBuffer cleared_output_buffer(cleared_output);
    sfFDN::AudioBuffer fresh_output_buffer(fresh_output);

    fdn->Process(input_buffer, warm_output_buffer);
    auto clone = fdn->Clone();
    fdn->Clear();
    fdn->Process(input_buffer, cleared_output_buffer);
    fresh->Process(input_buffer, fresh_output_buffer);
    for (size_t i = 0; i < fresh_output.size(); ++i)
    {
        REQUIRE(cleared_output[i] == Catch::Approx(fresh_output[i]));
    }

    std::array<float, kSampleCount> clone_output{};
    sfFDN::AudioBuffer clone_output_buffer(clone_output);
    clone->Process(input_buffer, clone_output_buffer);
    for (size_t i = 0; i < fresh_output.size(); ++i)
    {
        REQUIRE(clone_output[i] == Catch::Approx(fresh_output[i]));
    }
}

TEST_CASE("FDN rejects incompatible setters without replacing configured processors", "[fdn]")
{
    sfFDN::FDN fdn(4, 8);
    auto* const output_gains = fdn.GetOutputGains();
    auto* const feedback_matrix = fdn.GetFeedbackMatrix();

    auto wrong_output = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge);
    wrong_output->SetGains(std::array{1.f, 1.f, 1.f});
    REQUIRE_FALSE(fdn.SetOutputGains(std::move(wrong_output)));
    REQUIRE(fdn.GetOutputGains() == output_gains);

    auto wrong_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(
        sfFDN::ScalarFeedbackMatrixOptions{.source = sfFDN::GeneratedMatrixOptions{
                                               .matrix_size = 3,
                                               .generator = sfFDN::ScalarMatrixType::Identity,
                                           }});
    REQUIRE_FALSE(fdn.SetFeedbackMatrix(std::move(wrong_matrix)));
    REQUIRE(fdn.GetFeedbackMatrix() == feedback_matrix);

    REQUIRE_FALSE(fdn.SetDelays(std::array{8.f, 8.f, 8.f}));
    REQUIRE(fdn.GetDelayBank().InputChannelCount() == 4);
}

TEST_CASE("FDN SetOrder resets order-dependent components and preserves transpose", "[fdn]")
{
    sfFDN::FDN fdn(4, 8, true);
    fdn.SetLoopFilter(
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel, std::array{1.f, 1.f, 1.f, 1.f}));
    fdn.SetOrder(6);
    REQUIRE(fdn.GetOrder() == 6);
    REQUIRE(fdn.GetTranspose());
    REQUIRE(fdn.GetLoopFilter() == nullptr);
    REQUIRE(fdn.GetInputGains()->OutputChannelCount() == 6);
    REQUIRE(fdn.GetOutputGains()->InputChannelCount() == 6);

    auto* const input_gains = fdn.GetInputGains();
    auto* const output_gains = fdn.GetOutputGains();
    fdn.SetOrder(6);
    REQUIRE(fdn.GetInputGains() == input_gains);
    REQUIRE(fdn.GetOutputGains() == output_gains);

    fdn.SetTranspose(false);
    REQUIRE_FALSE(fdn.GetTranspose());
    fdn.SetOrder(3);
    REQUIRE(fdn.GetOrder() == 6);
}

TEST_CASE("FDN move operations preserve active processing state", "[fdn]")
{
    constexpr uint32_t kSampleCount = 32;
    std::array<float, kSampleCount> impulse{};
    impulse[0] = 1.F;
    std::array<float, kSampleCount> silence{};
    const sfFDN::AudioBuffer impulse_buffer(impulse);
    const sfFDN::AudioBuffer silence_buffer(silence);

    auto reference = CreatePyFDNGoldFDN();
    auto move_source = CreatePyFDNGoldFDN();
    auto assignment_source = CreatePyFDNGoldFDN();
    std::array<float, kSampleCount> warm_output{};
    sfFDN::AudioBuffer warm_output_buffer(warm_output);
    reference->Process(impulse_buffer, warm_output_buffer);
    std::ranges::fill(warm_output, 0.F);
    move_source->Process(impulse_buffer, warm_output_buffer);
    std::ranges::fill(warm_output, 0.F);
    assignment_source->Process(impulse_buffer, warm_output_buffer);

    std::array<float, kSampleCount> reference_output{};
    sfFDN::AudioBuffer reference_output_buffer(reference_output);
    reference->Process(silence_buffer, reference_output_buffer);

    sfFDN::FDN move_constructed(std::move(*move_source));
    std::array<float, kSampleCount> move_output{};
    sfFDN::AudioBuffer move_output_buffer(move_output);
    move_constructed.Process(silence_buffer, move_output_buffer);
    REQUIRE(move_output == reference_output);

    sfFDN::FDN move_assigned(4, 4);
    move_assigned = std::move(*assignment_source);
    std::array<float, kSampleCount> assignment_output{};
    sfFDN::AudioBuffer assignment_output_buffer(assignment_output);
    move_assigned.Process(silence_buffer, assignment_output_buffer);
    REQUIRE(assignment_output == reference_output);
}

TEST_CASE("FDN processing is allocation-free for normal, transposed, and configured networks", "[fdn]")
{
    constexpr uint32_t kBlockSize = 8;
    std::array<float, kBlockSize> input{};
    std::array<float, kBlockSize> output{};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    sfFDN::FDN normal(4, kBlockSize);
    sfFDN::FDN transposed(4, kBlockSize, true);
    sfFDN::FDNConfig config;
    config.fdn_size = 4;
    config.block_size = kBlockSize;
    config.sample_rate = 48000.f;
    config.delay_bank_config = {.delays = {16.f, 17.f, 19.f, 23.f}, .block_size = kBlockSize};
    config.input_block_config.parallel_gains_config = {.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                       .time_varying_config = {}};
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source = sfFDN::GeneratedMatrixOptions{
            .matrix_size = config.fdn_size,
            .generator = sfFDN::ScalarMatrixType::Hadamard,
        }};
    config.output_block_config.parallel_gains_config = {.gains = std::vector<float>(config.fdn_size, 0.5f),
                                                        .time_varying_config = {}};
    auto configured = sfFDN::CreateFDNFromConfig(config);

    normal.Process(input_buffer, output_buffer);
    transposed.Process(input_buffer, output_buffer);
    configured->Process(input_buffer, output_buffer);
    {
        sfFDNTest::ScopedAllocationCounter const allocation_counter;
        normal.Process(input_buffer, output_buffer);
        transposed.Process(input_buffer, output_buffer);
        configured->Process(input_buffer, output_buffer);
        REQUIRE(allocation_counter.Count() == 0);
    }
}

TEST_CASE("FDNConfig defaults are defined and reject an incomplete draft", "[fdn]")
{
    sfFDN::FDNConfig config;

    REQUIRE(config.fdn_size == 0U);
    REQUIRE_FALSE(config.transposed);
    REQUIRE(config.direct_gain == 0.F);
    REQUIRE(config.block_size == sfFDN::kDefaultBlockSize);
    REQUIRE(config.sample_rate == static_cast<float>(sfFDN::kDefaultSampleRate));
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
}

TEST_CASE("FDNConfig validates primary delay bank values and sizing", "[fdn]")
{
    SECTION("rejects malformed primary delays")
    {
        for (const float delay : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  0.F,
                                  -1.F,
                                  7.F,
                                  std::numeric_limits<float>::max()})
        {
            auto config = MakeFactoryConfig();
            config.delay_bank_config.delays[0] = delay;
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
        }
    }

    SECTION("rejects incompatible primary bank block sizes")
    {
        auto config = MakeFactoryConfig();
        config.delay_bank_config.block_size = 0;
        REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);

        config.delay_bank_config.block_size = config.block_size - 1U;
        REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
    }

    SECTION("rejects a primary bank with the wrong channel count")
    {
        auto config = MakeFactoryConfig();
        config.delay_bank_config.delays.pop_back();
        REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
    }

    SECTION("accepts boundary delays and a larger primary bank block size")
    {
        auto config = MakeFactoryConfig();
        config.delay_bank_config.block_size = config.block_size * 2U;
        const auto fdn = sfFDN::CreateFDNFromConfig(config);

        REQUIRE(fdn->GetDelayBank().GetDelays() == config.delay_bank_config.delays);
    }
}

TEST_CASE("FDNConfig accepts only documented attenuation bank cardinalities", "[fdn]")
{
    constexpr size_t kOrder = 4;

    SECTION("accepts shared and per-channel dedicated or loop attenuation")
    {
        for (const size_t count : {size_t{1}, kOrder})
        {
            auto dedicated = MakeFactoryConfig();
            dedicated.attenuation_filter_bank_config = MakeAttenuationBank(count);
            REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(dedicated));

            auto loop = MakeFactoryConfig();
            loop.loop_filter_configs.emplace_back(MakeAttenuationBank(count));
            REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(loop));
        }
    }

    SECTION("accepts exact-sized attenuation inserts")
    {
        auto input = MakeFactoryConfig();
        input.input_block_config.multichannel_processors.emplace_back(MakeAttenuationBank(kOrder));
        REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(input));

        auto output = MakeFactoryConfig();
        output.output_block_config.multichannel_processors.emplace_back(MakeAttenuationBank(kOrder));
        REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(output));
    }

    SECTION("rejects empty, short, and overlong banks in every placement")
    {
        for (const size_t count : {size_t{0}, size_t{2}, size_t{5}})
        {
            auto dedicated = MakeFactoryConfig();
            dedicated.attenuation_filter_bank_config = MakeAttenuationBank(count);
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(dedicated), std::runtime_error);

            auto loop = MakeFactoryConfig();
            loop.loop_filter_configs.emplace_back(MakeAttenuationBank(count));
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(loop), std::runtime_error);

            auto input = MakeFactoryConfig();
            input.input_block_config.multichannel_processors.emplace_back(MakeAttenuationBank(count));
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(input), std::runtime_error);

            auto output = MakeFactoryConfig();
            output.output_block_config.multichannel_processors.emplace_back(MakeAttenuationBank(count));
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(output), std::runtime_error);
        }
    }
}

TEST_CASE("FDNConfig permits short delay bank inserts", "[fdn]")
{
    const sfFDN::DelayBankOptions short_delays{
        .delays = {1.F, 2.F, 3.F, 4.F},
        .block_size = 4U,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };

    auto input = MakeFactoryConfig();
    input.input_block_config.multichannel_processors.emplace_back(short_delays);
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(input));

    auto loop = MakeFactoryConfig();
    loop.loop_filter_configs.emplace_back(short_delays);
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(loop));

    auto output = MakeFactoryConfig();
    output.output_block_config.multichannel_processors.emplace_back(short_delays);
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(output));
}

TEST_CASE("FDNConfig rejects overflowing delay bank inserts", "[fdn]")
{
    const auto reject_in_all_placements = [](const sfFDN::DelayBankOptions& delay_bank) {
        for (const uint32_t placement : {0U, 1U, 2U})
        {
            auto config = MakeFactoryConfig();
            if (placement == 0U)
            {
                config.input_block_config.multichannel_processors.emplace_back(delay_bank);
            }
            else if (placement == 1U)
            {
                config.loop_filter_configs.emplace_back(delay_bank);
            }
            else
            {
                config.output_block_config.multichannel_processors.emplace_back(delay_bank);
            }
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
        }
    };

    reject_in_all_placements({
        .delays = {0.F, 0.F, 0.F, 0.F},
        .block_size = 2147483616U,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    });
    reject_in_all_placements({
        .delays = {1.F, 1.F, 1.F, 1.F},
        .block_size = 2147483584U,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    });
}

TEST_CASE("FDNConfig constructs ordered tone correction paths", "[fdn]")
{
    for (const bool transposed : {false, true})
    {
        const auto empty = MakeFactoryConfig(transposed);
        const auto empty_fdn = sfFDN::CreateFDNFromConfig(empty);
        REQUIRE(empty_fdn->GetTCFilter() == nullptr);
        const auto baseline = RenderFactoryConfig(empty);
        REQUIRE(std::ranges::any_of(baseline, [](float sample) { return sample != 0.F; }));

        auto single = MakeFactoryConfig(transposed);
        single.tone_correction_filters.emplace_back(sfFDN::FirOptions{.coeffs = {2.F}});
        const auto single_fdn = sfFDN::CreateFDNFromConfig(single);
        REQUIRE(dynamic_cast<sfFDN::AudioProcessorChain*>(single_fdn->GetTCFilter()) == nullptr);
        const auto single_output = RenderFactoryConfig(single);

        auto multiple = MakeFactoryConfig(transposed);
        multiple.tone_correction_filters = {
            sfFDN::ControllableFullWaveRectifierOptions{.alpha = 1.F, .antialiasing = false, .dc_block = false},
            sfFDN::FirOptions{.coeffs = {-1.F}},
        };
        const auto multiple_fdn = sfFDN::CreateFDNFromConfig(multiple);
        const auto* chain = dynamic_cast<sfFDN::AudioProcessorChain*>(multiple_fdn->GetTCFilter());
        REQUIRE(chain != nullptr);
        REQUIRE(chain->GetProcessorCount() == 2U);
        const auto multiple_output = RenderFactoryConfig(multiple);

        for (size_t index = 0; index < baseline.size(); ++index)
        {
            REQUIRE(single_output[index] == Catch::Approx(2.F * baseline[index]));
            REQUIRE(multiple_output[index] == Catch::Approx(-std::abs(baseline[index])));
        }
    }
}
