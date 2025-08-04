#include "doctest.h"

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/QR>

#include "sffdn/sffdn.h"

#include "matrix_multiplication.h"
#include "test_utils.h"

TEST_CASE("VelvetFFM")
{
    constexpr size_t num_stages = 2;
    constexpr float sparsity = 2.f;
    constexpr size_t N = 4;
    constexpr float cascade_gain = 1.f;

    sfFDN::CascadedFeedbackMatrixInfo ffm_info = sfFDN::ConstructCascadedFeedbackMatrix(
        N, num_stages, sparsity, sfFDN::ScalarMatrixType::Hadamard, cascade_gain);

    auto ffm = sfFDN::MakeFilterFeedbackMatrix(ffm_info);
    CHECK(ffm != nullptr);
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

TEST_CASE("Inplace" * doctest::skip(true))
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
    constexpr uint32_t delays[] = {11, 11, 2, 6, 10, 14, 17, 8, 2, 6, 19, 5, 10, 19, 1, 13};
    sfFDN::ScalarFeedbackMatrix mixing_matrix = sfFDN::ScalarFeedbackMatrix::Hadamard(N);
    sfFDN::DelayMatrix delay_matrix(4, delays, mixing_matrix);

    constexpr size_t kBlockSize = 32;
    std::array<float, N * kBlockSize> input = {0.f};
    std::array<float, N * kBlockSize> output = {0.f};

    for (auto i = 0; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f; // Set the first sample of each channel to 1
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());
    delay_matrix.Process(input_buffer, output_buffer);

    const std::array<float, kBlockSize> expected_output_ch1 = {0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.5, 0, 0, 0, 0,
                                                               0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch2 = {
        0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch3 = {0, -0.5, 0.5, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0, 0.5,  0,   -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch4 = {
        0, 0, 0, 0, 0, -0.5, 0.5, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (auto i = 0; i < output_buffer.SampleCount(); ++i)
    {
        CHECK(output_buffer.GetChannelSpan(0)[i] == doctest::Approx(expected_output_ch1[i]));
        CHECK(output_buffer.GetChannelSpan(1)[i] == doctest::Approx(expected_output_ch2[i]));
        CHECK(output_buffer.GetChannelSpan(2)[i] == doctest::Approx(expected_output_ch3[i]));
        CHECK(output_buffer.GetChannelSpan(3)[i] == doctest::Approx(expected_output_ch4[i]));
    }
}

TEST_CASE("FilterFeedbackMatrix")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t K = 1;

    std::array<uint32_t, N*(K + 1)> delays = {0, 5, 6, 11, 0, 12, 24, 36};
    std::vector<sfFDN::ScalarFeedbackMatrix> mixing_matrices;
    for (uint32_t i = 0; i < K; ++i)
    {
        mixing_matrices.push_back(sfFDN::ScalarFeedbackMatrix::Hadamard(N));
    }

    // sfFDN::FilterFeedbackMatrix ffm(N);
    // ffm.ConstructMatrix(delays, mixing_matrices);

    auto ffm = CreateFFM(N, K, 3);

    constexpr size_t kBlockSize = 64;
    std::array<float, N * kBlockSize> input = {0.f};
    // input[0] = 1.f;

    for (uint32_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::array<float, N * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    ffm->Process(input_buffer, output_buffer);

    // for (size_t i = 0; i < kBlockSize; ++i)
    // {
    //     std::print("{} \t", i + 1);
    //     for (size_t j = 0; j < N; ++j)
    //     {
    //         std::print("{} \t", output_buffer.GetChannelSpan(j)[i]);
    //     }
    //     std::print("\n");
    // }
}