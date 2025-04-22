#include "doctest.h"

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/QR>

#include "delay_matrix.h"
#include "filter_feedback_matrix.h"
#include "mixing_matrix.h"

#include "test_utils.h"

TEST_CASE("Identity matrix")
{
    constexpr size_t N = 4;
    fdn::MixMat mix_mat(N);

    std::array<float, N * 2> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * 2> output;

    mix_mat.Tick(input, output);

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(input[i] == output[i]);
        CHECK(input[i + 1] == output[i + 1]);
        CHECK(input[i + 2] == output[i + 2]);
        CHECK(input[i + 3] == output[i + 3]);
    }
}

TEST_CASE("Inplace")
{
    constexpr size_t N = 4;
    fdn::MixMat mix_mat(N);

    std::array<float, N * 2> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * 2> output = input;

    mix_mat.Tick(output, output);

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(input[i] == output[i]);
        CHECK(input[i + 1] == output[i + 1]);
        CHECK(input[i + 2] == output[i + 2]);
        CHECK(input[i + 3] == output[i + 3]);
    }
}

TEST_CASE("Householder")
{
    constexpr size_t N = 4;
    auto mix_mat = fdn::MixMat::Householder(N);

    std::array<float, N * 2> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * 2> output;

    mix_mat.Tick(input, output);

    constexpr std::array<float, N * N> expected = {-4, -3, -2, -1, -8, -7, -6, -5};

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(expected[i] == output[i]);
        CHECK(expected[i + 1] == output[i + 1]);
        CHECK(expected[i + 2] == output[i + 2]);
        CHECK(expected[i + 3] == output[i + 3]);
    }
}

TEST_CASE("Householder2")
{
    constexpr size_t N = 4;
    std::array<float, N> v = {1, 2, 3, 4};
    auto mix_mat = fdn::MixMat::Householder(v);

    mix_mat.Print();
}

TEST_CASE("Hadamard")
{
    constexpr std::array<size_t, 4> order = {2, 4, 8, 16};

    for (size_t i = 0; i < order.size(); ++i)
    {
        const size_t N = order[i];
        auto mix_mat = fdn::MixMat::Hadamard(N);

        mix_mat.Print();
        std::cout << std::endl;
    }
}

TEST_CASE("MatrixAssignment")
{
    constexpr size_t N = 4;
    fdn::MixMat mix_mat(N);

    std::array<float, N * N> matrix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    mix_mat.SetMatrix(matrix);

    std::array<float, N * 2> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * 2> output = {0.f};
    mix_mat.Tick(input, output);

    for (size_t i = 0; i < N; ++i)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    mix_mat.Print();
}

TEST_CASE("DelayMatrix")
{
    constexpr size_t N = 4;
    constexpr size_t delays[] = {0, 1, 3, 5};
    fdn::DelayMatrix delay_matrix(4, delays);

    auto mix_mat = fdn::MixMat::Householder(N);
    delay_matrix.SetMatrix(mix_mat);

    std::array<float, N * 8> input = {0.f};

    for (size_t i = 0; i < N; ++i)
    {
        input[i] = 1.f;
    }

    std::array<float, N * 8> output = {0.f};
    delay_matrix.Tick(input, output);

    for (size_t i = 0; i < output.size(); i += N)
    {
        for (size_t j = 0; j < N; ++j)
        {
            std::cout << output[i + j] << " ";
        }
        std::cout << std::endl;
    }

    float energy_in = 0.f;
    for (size_t i = 0; i < input.size(); ++i)
    {
        energy_in += input[i] * input[i];
    }

    float energy_out = 0.f;
    for (size_t i = 0; i < output.size(); ++i)
    {
        energy_out += output[i] * output[i];
    }

    CHECK(energy_in == doctest::Approx(energy_out).epsilon(0.01));
}

TEST_CASE("FilterFeedbackMatrix")
{
    constexpr size_t N = 4;
    constexpr size_t K = 4;
    std::array<size_t, N*(K - 1)> delays = {0};
    for (size_t i = 0; i < N * (K - 1); ++i)
    {
        delays[i] = i % N;
    }

    fdn::FilterFeedbackMatrix filter_feedback_matrix(N, K);
    filter_feedback_matrix.SetDelays(delays);

    std::vector<fdn::MixMat> mixing_matrices(K);
    for (size_t i = 0; i < K; ++i)
    {
        mixing_matrices[i] = fdn::MixMat::Householder(N);
    }
    // filter_feedback_matrix.SetMatrices(mixing_matrices);

    filter_feedback_matrix.DumpDelays();

    constexpr size_t ITER = 16;
    std::array<float, N * ITER> input = {0.f};

    for (size_t i = 0; i < N; ++i)
    {
        input[i] = 1.f;
    }

    std::array<float, N * ITER> output = {0.f};
    filter_feedback_matrix.Tick(input, output);
    for (size_t i = 0; i < output.size(); i += N)
    {
        for (size_t j = 0; j < N; ++j)
        {
            std::cout << output[i + j] << " ";
        }
        std::cout << std::endl;
    }

    float energy_in = 0.f;
    for (size_t i = 0; i < input.size(); ++i)
    {
        energy_in += input[i] * input[i];
    }

    float energy_out = 0.f;
    for (size_t i = 0; i < output.size(); ++i)
    {
        energy_out += output[i] * output[i];
    }

    CHECK(energy_in == doctest::Approx(energy_out).epsilon(0.01));
}

TEST_CASE("FFM_Hadamard")
{
    constexpr size_t N = 4;
    constexpr size_t K = 4;
    std::array<size_t, N*(K - 1)> delays = {1, 5, 6, 10, 0, 12, 24, 36, 0, 48, 96, 144};

    std::vector<fdn::MixMat> mixing_matrices(K);
    for (size_t i = 0; i < K; ++i)
    {
        mixing_matrices[i] = fdn::MixMat::Hadamard(N);
    }

    fdn::FilterFeedbackMatrix filter_feedback_matrix(N, K);
    filter_feedback_matrix.SetDelays(delays);
    filter_feedback_matrix.SetMatrices(mixing_matrices);

    constexpr size_t ITER = 16;
    std::array<float, N * ITER> input = {0.f};

    for (size_t i = 0; i < 1; ++i)
    {
        input[i] = 1.f;
    }
    std::array<float, N * ITER> output = {0.f};

    filter_feedback_matrix.Tick(input, output);

    // for (size_t i = 0; i < output.size(); i += N)
    // {
    //     for (size_t j = 0; j < N; ++j)
    //     {
    //         std::cout << output[i + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}