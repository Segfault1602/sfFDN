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
    constexpr size_t ITER = 48000;
    constexpr size_t N = 6;
    constexpr size_t block_size = 512;
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
    filter->dump_coeffs();
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
    constexpr size_t ITER = 48000;
    constexpr size_t N = 6;
    constexpr size_t block_size = 512;
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
    filter->dump_coeffs();
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

TEST_CASE("FDN_FFM")
{
    constexpr size_t SR = 48000;
    constexpr float SIMULATION_TIME = 0.01;
    constexpr size_t ITER = 20000;
    constexpr size_t N = 6;
    constexpr size_t block_size = 512;
    // constexpr std::array<float, N> input_gains = {0.072116069, 0.24890353,   0.97228086,
    //   -0.38236806, -0.057921566, -0.39115807};
    // constexpr std::array<float, N> output_gains = {-0.4632, -0.3661, 0.3090, 0.3014, -0.4920, 0.5870};
    constexpr std::array<float, N> delays = {593, 743, 929, 1153, 1399, 1699};

    constexpr std::array<float, N> input_gains = {1, 1, 1, 1, 1, 1};
    constexpr std::array<float, N> output_gains = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    // constexpr std::array<float, N> delays = {3, 5, 7, 11, 13, 17};

    constexpr std::array<float, N * N> mixing_matrix = {0.f};

    fdn::FDN fdn(N, block_size, true);
    fdn.SetInputGains(input_gains);
    fdn.SetOutputGains(output_gains);
    fdn.SetDirectGain(0.f);
    fdn.GetDelayBank()->SetDelays(delays);

    constexpr size_t K = 4;
    std::array<size_t, N*(K - 1)> ffm_delays = {
        2, 3, 8, 10, 14, 16, 0, 18, 36, 54, 72, 90, 0, 108, 216, 324, 432, 540,
    };

    auto ffm = std::make_unique<fdn::FilterFeedbackMatrix>(N, K);
    ffm->SetDelays(ffm_delays);

    std::vector<fdn::MixMat> mixing_matrices(K);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    for (size_t i = 0; i < K; ++i)
    {
        float u_n[N] = {0.f};
        for (size_t j = 0; j < N; ++j)
        {
            u_n[j] = dis(gen);
        }

        mixing_matrices[i] = fdn::MixMat::Householder(u_n);
    }

    ffm->SetMatrices(mixing_matrices);

    fdn.SetFeedbackMatrix(std::move(ffm));

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
    // filter->dump_coeffs();
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

    constexpr const char* output_filename = "test_ffm.wav";
    SF_INFO sfinfo;
    sfinfo.channels = 1;
    sfinfo.samplerate = SR;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sfinfo.frames = output.size();

    SNDFILE* outfile = sf_open(output_filename, SFM_WRITE, &sfinfo);
    if (outfile == nullptr)
    {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }

    sf_write_float(outfile, output.data(), output.size());
    sf_close(outfile);
}

TEST_CASE("Mixing Matrix" * doctest::skip(true))
{
    const size_t N = 4;
    const size_t col = 4;
    const size_t row = 16;

    std::vector<float> feedback = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> feedback_map(feedback.data(), row,
                                                                                                   col);

    std::cout << feedback_map << std::endl;
    std::cout << "----" << std::endl;

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::MatrixXf v = Eigen::VectorXf::Ones(N);

    Eigen::MatrixXf H = I - (2.f / N) * (v * v.transpose());

    auto mix_mat = fdn::MixMat::Householder(4);

    // feedback_map = feedback_map * H;
    mix_mat.Tick(feedback, feedback);

    std::cout << feedback_map << std::endl;
}

TEST_CASE("FDN_IR")
{
    constexpr size_t SR = 48000;
    constexpr float SIMULATION_TIME = 0.01;
    constexpr size_t ITER = SR * 4;
    constexpr size_t N = 6;
    constexpr size_t block_size = 512;
    constexpr std::array<float, N> input_gains = {
        0.457157433032990,  1.162811994552612,  -1.268344402313232,
        -0.104579553008080, -0.765608072280884, 0.270698904991150,
    };
    constexpr std::array<float, N> output_gains = {
        -0.027818815782666, 0.186146244406700,  0.260880768299103,
        0.038124702870846,  -0.046348180621862, -0.002050762297586,
    };
    constexpr std::array<float, N> delays = {887, 911, 941, 1699, 1951, 2053};

    constexpr float t60_dc = 2.0f;
    constexpr float t60_ny = 1.0f;

    fdn::FDN fdn(N, block_size, true);
    fdn.SetInputGains(input_gains);
    fdn.SetOutputGains(output_gains);

    // // clang-format off
    // constexpr std::array<float, N * N> matrix =
    // {  0.11841482, -0.08641767, -0.06865135, -0.45989491, -0.19026373,  0.85211108,
    //   -0.40942193, -0.25128215, -0.51994006,  0.52926089, -0.43227318,  0.17865077,
    //   -0.63321982, -0.14494175,  0.09990028, -0.60851222, -0.31211553, -0.31676688,
    //    0.05428409,  0.62807138, -0.71134088, -0.27417947,  0.04600297, -0.13886351,
    //    0.1574579,  -0.69526201, -0.45436149, -0.22953005,  0.45828363, -0.1505505,
    //   -0.62421513,  0.17469468,  0.05019005,  0.10116337,  0.68365441,  0.31575437,
    // };
    // // clang-format on

    // auto mix_mat = std::make_unique<fdn::MixMat>(N);
    // mix_mat->SetMatrix(matrix);
    // fdn.SetFeedbackMatrix(std::move(mix_mat));

    auto ffm = CreateFFM(N, 4, 3);
    fdn.SetFeedbackMatrix(std::move(ffm));

    auto filter_bank = fdn.GetFilterBank();
    for (size_t i = 0; i < N; i++)
    {
        auto sos = kAbsorbtionSOS[i];
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

    auto delay_bank = fdn.GetDelayBank();
    delay_bank->SetDelays(delays);

    std::unique_ptr<fdn::SchroederAllpassSection> schroeder_section = std::make_unique<fdn::SchroederAllpassSection>(N);

    std::vector<size_t> schroeder_delays = {2, 3, 5, 7, 11, 13};
    schroeder_section->SetDelays(schroeder_delays);
    std::vector<float> schroeder_gains = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f};
    schroeder_section->SetGains(schroeder_gains);
    fdn.SetSchroederSection(std::move(schroeder_section));

    std::vector<float> input(ITER, 0.f);
    std::vector<float> output(ITER, 0.f);

    input[0] = 1.f;

    for (size_t i = 0; i < input.size(); i += block_size)
    {
        std::span<float> input_span{input.data() + i, block_size};
        std::span<float> output_span{output.data() + i, block_size};
        fdn.Tick(input_span, output_span);
    }

    constexpr const char* output_filename = "fdn_ir_test_output.wav";
    SF_INFO sfinfo;
    sfinfo.channels = 1;
    sfinfo.samplerate = SR;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    sfinfo.frames = output.size();

    SNDFILE* outfile = sf_open(output_filename, SFM_WRITE, &sfinfo);
    if (outfile == nullptr)
    {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }

    sf_write_float(outfile, output.data(), output.size());
    sf_close(outfile);
}
