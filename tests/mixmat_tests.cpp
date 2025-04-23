#include "doctest.h"

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/QR>

#include "delay_matrix.h"
#include "filter_feedback_matrix.h"
#include "mixing_matrix.h"

#include "test_utils.h"

TEST_CASE("IdentityMatrix")
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

TEST_CASE("Householder")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    auto mix_mat = fdn::MixMat::Householder(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    mix_mat.Tick(input, output);

    // clang-format off
    constexpr std::array<float, N * kBlockSize> expected = {
         0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(expected[i] == doctest::Approx(output[i]).epsilon(0.01));
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

TEST_CASE("Inplace")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    auto mix_mat = fdn::MixMat::Householder(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    mix_mat.Tick(input, input);

    // clang-format off
    constexpr std::array<float, N * kBlockSize> expected = {
         0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(expected[i] == doctest::Approx(input[i]).epsilon(0.01));
    }
}

TEST_CASE("Hadamard")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    auto mix_mat = fdn::MixMat::Hadamard(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    mix_mat.Tick(input, output);

    // clang-format off
    constexpr std::array<float, N * kBlockSize> expected = {
        0.5000,  0.5000,  0.5000,  0.5000,  0, 0, 0, 0,
        0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (size_t i = 0; i < input.size(); i += N)
    {
        CHECK(expected[i] == doctest::Approx(output[i]).epsilon(0.01));
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
    auto ffm = CreateFFM(N, K, 0);

    constexpr size_t ITER = 16;
    std::array<float, N * ITER> input = {0.f};

    for (size_t i = 0; i < N; ++i)
    {
        input[i] = 1.f;
    }

    std::array<float, N * ITER> output = {0.f};
    ffm->Tick(input, output);

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