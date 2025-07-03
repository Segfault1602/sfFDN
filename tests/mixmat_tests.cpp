#include "doctest.h"

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/QR>

#include "delay_matrix.h"
#include "feedback_matrix.h"
#include "filter_feedback_matrix.h"
#include "matrix_gallery.h"
#include "matrix_multiplication.h"

#include "test_utils.h"

TEST_CASE("VelvetFFM")
{
    auto vffm = sfFDN::ConstructCascadedFeedbackMatrix(4, 4, 4, sfFDN::ScalarMatrixType::Hadamard, 0.99);

    std::cout << "Velvet FFM: " << vffm.N << "x" << vffm.K << std::endl;
    std::cout << "Delays: ";

    for (size_t i = 0; i < vffm.delays.size(); ++i)
    {
        std::cout << vffm.delays[i] << " ";

        if ((i + 1) % vffm.N == 0)
        {
            std::cout << std::endl;
        }
    }

    std::cout << "Matrices: " << std::endl;
    for (size_t i = 0; i < vffm.matrices.size(); ++i)
    {
        std::cout << vffm.matrices[i] << " ";
        if ((i + 1) % (vffm.N) == 0)
        {
            std::cout << std::endl;
        }

        if ((i + 1) % (vffm.N * vffm.N) == 0)
        {
            std::cout << std::endl;
        }
    }
}

TEST_CASE("IdentityMatrix")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat(N);

    std::array<float, N * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * kBlockSize> output;

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    mix_mat.Process(input_buffer, output_buffer);

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
    auto mix_mat = sfFDN::ScalarFeedbackMatrix::Householder(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    mix_mat.Process(input_buffer, output_buffer);

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

TEST_CASE("Hadamard")
{
    SUBCASE("Hadamard_4")
    {
        constexpr size_t N = 4;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix::Hadamard(N);

        std::array<float, N> input = {1, 2, 3, 4};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input.data());
        sfFDN::AudioBuffer output_buffer(1, N, output.data());

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {5, -1, -2, 0};

        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }
    }

    SUBCASE("Hadamard_8")
    {
        constexpr size_t N = 8;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix::Hadamard(N);

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input.data());
        sfFDN::AudioBuffer output_buffer(1, N, output.data());

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {
            12.727922061357855, -1.414213562373095, -2.828427124746190, 0, -5.656854249492380, 0, 0, 0};

        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }
    }

    SUBCASE("Hadamard_16")
    {
        constexpr size_t N = 16;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix::Hadamard(N);

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input.data());
        sfFDN::AudioBuffer output_buffer(1, N, output.data());

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }
    }
}

TEST_CASE("Inplace")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix::Householder(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());

    mix_mat.Process(input_buffer, input_buffer);

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

TEST_CASE("Hadamard_Block")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix::Hadamard(N);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    mix_mat.Process(input_buffer, output_buffer);

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
    constexpr size_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat(N);

    std::array<float, N * N> matrix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    mix_mat.SetMatrix(matrix);

    std::array<float, N * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    mix_mat.Process(input_buffer, output_buffer);
}

TEST_CASE("DelayMatrix")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t delays[] = {0, 1, 3, 5};
    sfFDN::DelayMatrix delay_matrix(4, delays);

    auto mix_mat = sfFDN::ScalarFeedbackMatrix::Householder(N);
    delay_matrix.SetMatrix(mix_mat);

    std::array<float, N * 8> input = {0.f};

    for (uint32_t i = 0; i < N; ++i)
    {
        input[i] = 1.f;
    }

    constexpr uint32_t kBlockSize = 8;
    std::array<float, N * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    delay_matrix.Process(input_buffer, output_buffer);

    float energy_in = 0.f;
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        energy_in += input[i] * input[i];
    }

    float energy_out = 0.f;
    for (uint32_t i = 0; i < output.size(); ++i)
    {
        energy_out += output[i] * output[i];
    }

    CHECK(energy_in == doctest::Approx(energy_out));
}

TEST_CASE("FilterFeedbackMatrix")
{
    constexpr uint32_t N = 8;
    constexpr uint32_t K = 4;
    auto ffm = CreateFFM(N, K, 1);

    constexpr size_t ITER = 1024;
    std::array<float, N * ITER> input = {0.f};

    for (uint32_t i = 0; i < N; ++i)
    {
        input[i] = 1.f;
    }

    std::array<float, N * ITER> output = {0.f};

    sfFDN::AudioBuffer input_buffer(ITER, N, input.data());
    sfFDN::AudioBuffer output_buffer(ITER, N, output.data());

    ffm->Process(input_buffer, output_buffer);

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