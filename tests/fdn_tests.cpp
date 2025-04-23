#include "doctest.h"

#include <fstream>
#include <iostream>
#include <random>
#include <span>

#include <Eigen/Core>
#include <sndfile.h>

#include "fdn.h"
#include "filter_coeffs.h"
#include "filter_design.h"
#include "filter_feedback_matrix.h"
#include "mixing_matrix.h"

#include "test_utils.h"

TEST_CASE("FDN")
{
    constexpr size_t SR = 48000;
    constexpr float SIMULATION_TIME = 0.01;
    constexpr size_t block_size = 8;
    constexpr size_t ITER = ((SR / block_size) + 1) * block_size;
    constexpr size_t N = 6;
    constexpr std::array<float, N> input_gains = {0.072116069, 0.24890353,   0.97228086,
                                                  -0.38236806, -0.057921566, -0.39115807};
    constexpr std::array<float, N> output_gains = {-0.46316639, -0.36613876, 0.30902779,
                                                   0.30143532,  -0.49200505, 0.58704174};
    constexpr std::array<float, N> delays = {593, 743, 929, 1153, 1399, 1699};

    constexpr std::array<float, N * N> mixing_matrix = {
        0.590748429298401,  0.457586556673050,  0.0557801127433777, -0.148047655820847,  -0.478258520364761,
        -0.433439940214157, -0.158531382679939, 0.433001756668091,  -0.0591235160827637, 0.626041889190674,
        0.430089294910431,  -0.454946815967560, -0.665803074836731, 0.195845842361450,   0.568070054054260,
        -0.251500934362412, -0.263658404350281, -0.250756144523621, 0.239477828145027,   -0.236257210373878,
        0.618841290473938,  0.622415661811829,  -0.255638062953949, 0.226088821887970,   0.266185045242310,
        -0.500568747520447, 0.346136510372162,  -0.255272954702377, 0.454669415950775,   -0.535609304904938,
        0.233208581805229,  0.508312821388245,  0.409773439168930,  -0.265208065509796,  0.494672924280167,
        0.451974451541901};

    fdn::FDN fdn(N, block_size);
    fdn.SetInputGains(input_gains);
    fdn.SetOutputGains(output_gains);
    fdn.SetDirectGain(0.f);
    fdn.GetDelayBank()->SetDelays(delays);

    auto mix_mat = std::make_unique<fdn::MixMat>(N);
    mix_mat->SetMatrix(mixing_matrix);

    fdn.SetFeedbackMatrix(std::move(mix_mat));

    fdn.SetBypassAbsorption(false);

    auto filter_bank = fdn.GetFilterBank();
    for (size_t i = 0; i < N; i++)
    {
        auto sos = k_h001_AbsorbtionSOS[i];
        fdn::CascadedBiquads* filter = new fdn::CascadedBiquads();

        std::vector<float> coeffs;
        for (size_t j = 0; j < sos.size(); j++)
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

        filter_bank->SetFilter(i, filter);
    }

    std::vector<float> coeffs;
    for (size_t i = 0; i < k_h001_EqualizationSOS.size(); i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<fdn::CascadedBiquads> filter = std::make_unique<fdn::CascadedBiquads>();
    filter->SetCoefficients(k_h001_EqualizationSOS.size(), coeffs);
    fdn.SetTCFilter(std::move(filter));

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);

    input[0] = 1.f;

    for (size_t i = 0; i < input.size(); i += block_size)
    {
        std::span<float> input_span{input.data() + i, block_size};
        std::span<float> output_span{output.data() + i, block_size};
        fdn.Tick(input_span, output_span);
    }

    {
        constexpr const char* expected_output_filename = "./tests/fdn_gold_test.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        CHECK(expected_output_file != nullptr);

        CHECK(sfinfo.channels == 1);
        CHECK(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        CHECK(read == sfinfo.frames);
        sf_close(expected_output_file);

        for (size_t i = 0; i < 1000; ++i)
        {
            CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(5e-4));
        }
    }
}

TEST_CASE("FDN_Transposed")
{
    constexpr size_t SR = 48000;
    constexpr float SIMULATION_TIME = 0.01;
    constexpr size_t N = 6;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = ((SR / block_size) + 1) * block_size;
    constexpr std::array<float, N> input_gains = {0.072116069, 0.24890353,   0.97228086,
                                                  -0.38236806, -0.057921566, -0.39115807};
    constexpr std::array<float, N> output_gains = {-0.46316639, -0.36613876, 0.30902779,
                                                   0.30143532,  -0.49200505, 0.58704174};
    constexpr std::array<float, N> delays = {593, 743, 929, 1153, 1399, 1699};

    constexpr std::array<float, N * N> mixing_matrix = {
        0.590748429298401,  0.457586556673050,  0.0557801127433777, -0.148047655820847,  -0.478258520364761,
        -0.433439940214157, -0.158531382679939, 0.433001756668091,  -0.0591235160827637, 0.626041889190674,
        0.430089294910431,  -0.454946815967560, -0.665803074836731, 0.195845842361450,   0.568070054054260,
        -0.251500934362412, -0.263658404350281, -0.250756144523621, 0.239477828145027,   -0.236257210373878,
        0.618841290473938,  0.622415661811829,  -0.255638062953949, 0.226088821887970,   0.266185045242310,
        -0.500568747520447, 0.346136510372162,  -0.255272954702377, 0.454669415950775,   -0.535609304904938,
        0.233208581805229,  0.508312821388245,  0.409773439168930,  -0.265208065509796,  0.494672924280167,
        0.451974451541901};

    fdn::FDN fdn(N, block_size, true);
    fdn.SetInputGains(input_gains);
    fdn.SetOutputGains(output_gains);
    fdn.SetDirectGain(0.f);
    fdn.GetDelayBank()->SetDelays(delays);

    auto mix_mat = std::make_unique<fdn::MixMat>(N);
    mix_mat->SetMatrix(mixing_matrix);

    fdn.SetFeedbackMatrix(std::move(mix_mat));

    fdn.SetBypassAbsorption(false);

    auto filter_bank = fdn.GetFilterBank();
    for (size_t i = 0; i < N; i++)
    {
        auto sos = k_h001_AbsorbtionSOS[i];
        fdn::CascadedBiquads* filter = new fdn::CascadedBiquads();

        std::vector<float> coeffs;
        for (size_t j = 0; j < sos.size(); j++)
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

        filter_bank->SetFilter(i, filter);
    }

    std::vector<float> coeffs;
    for (size_t i = 0; i < k_h001_EqualizationSOS.size(); i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<fdn::CascadedBiquads> filter = std::make_unique<fdn::CascadedBiquads>();
    filter->SetCoefficients(k_h001_EqualizationSOS.size(), coeffs);
    fdn.SetTCFilter(std::move(filter));

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);

    input[0] = 1.f;

    for (size_t i = 0; i < input.size(); i += block_size)
    {
        std::span<float> input_span{input.data() + i, block_size};
        std::span<float> output_span{output.data() + i, block_size};
        fdn.Tick(input_span, output_span);
    }

    {
        constexpr const char* expected_output_filename = "./tests/fdn_gold_test_transposed.wav";
        SF_INFO sfinfo;
        SNDFILE* expected_output_file = sf_open(expected_output_filename, SFM_READ, &sfinfo);

        CHECK(expected_output_file != nullptr);

        CHECK(sfinfo.channels == 1);
        CHECK(sfinfo.samplerate == SR);

        std::vector<float> expected_output(sfinfo.frames);
        sf_count_t read = sf_readf_float(expected_output_file, expected_output.data(), sfinfo.frames);
        CHECK(read == sfinfo.frames);
        sf_close(expected_output_file);

        for (size_t i = 0; i < 1000; ++i)
        {
            CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(5e-4));
        }
    }
}