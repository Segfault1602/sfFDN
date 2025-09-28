#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <limits>
#include <ranges>

#include "sffdn/audio_buffer.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/sffdn.h"

#include "matrix_multiplication.h"
#include "test_utils.h"

TEST_CASE("VelvetFFM")
{
    constexpr uint32_t num_stages = 2;
    constexpr float sparsity = 2.f;
    constexpr uint32_t N = 4;
    constexpr float cascade_gain = 1.f;

    sfFDN::CascadedFeedbackMatrixInfo ffm_info = sfFDN::ConstructCascadedFeedbackMatrix(
        N, num_stages, sparsity, sfFDN::ScalarMatrixType::Hadamard, cascade_gain);

    auto ffm = sfFDN::MakeFilterFeedbackMatrix(ffm_info);
    REQUIRE(ffm != nullptr);
}

TEST_CASE("VariableDiffusionMatrix")
{
    constexpr uint32_t N = 8;
    auto mat = sfFDN::GenerateMatrix(N, sfFDN::ScalarMatrixType::VariableDiffusion, 0.f, 1.0f);

    for (auto i = 0u; i < N; ++i)
    {
        for (auto j = 0u; j < N; ++j)
        {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}

TEST_CASE("IdentityMatrix")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat(N);

    std::array<float, N * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * kBlockSize> output{};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    mix_mat.Process(input_buffer, output_buffer);

    for (auto [in, out] : std::views::zip(input, output))
    {
        REQUIRE(in == out);
    }

    float energy_in = 0.f;
    for (auto in : input)
    {
        energy_in += in * in;
    }

    float energy_out = 0.f;
    for (auto out : output)
    {
        energy_out += out * out;
    }

    REQUIRE_THAT(energy_in, Catch::Matchers::WithinAbs(energy_out, std::numeric_limits<float>::epsilon()));
}

TEST_CASE("Householder")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Householder);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (auto i = 0u; i < N; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    mix_mat.Process(input_buffer, output_buffer);

    // clang-format off
    constexpr std::array<float, N * kBlockSize> expected = {
         0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (auto i = 0u; i < input.size(); i += N)
    {
        REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
    }

    float energy_in = 0.f;
    for (auto i = 0u; i < input.size(); ++i)
    {
        energy_in += input[i] * input[i];
    }

    float energy_out = 0.f;
    for (auto i = 0u; i < output.size(); ++i)
    {
        energy_out += output[i] * output[i];
    }

    REQUIRE_THAT(energy_in, Catch::Matchers::WithinAbs(energy_out, std::numeric_limits<float>::epsilon()));
}

TEST_CASE("FeedbackMatrixHadamard")
{
    SECTION("Hadamard_4")
    {
        constexpr uint32_t N = 4;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard);

        std::array<float, N> input = {1, 2, 3, 4};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input);
        sfFDN::AudioBuffer output_buffer(1, N, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {5, -1, -2, 0};

        for (auto i = 0u; i < input.size(); i += N)
        {
            REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }
    }

    SECTION("Hadamard_8")
    {
        constexpr uint32_t N = 8;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard);

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input);
        sfFDN::AudioBuffer output_buffer(1, N, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {
            12.727922061357855, -1.414213562373095, -2.828427124746190, 0, -5.656854249492380, 0, 0, 0};

        for (auto i = 0u; i < input.size(); i += N)
        {
            REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }
    }

    SECTION("Hadamard_16")
    {
        constexpr uint32_t N = 16;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard);

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, N> output;

        sfFDN::AudioBuffer input_buffer(1, N, input);
        sfFDN::AudioBuffer output_buffer(1, N, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, N> expected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        for (auto i = 0u; i < input.size(); i += N)
        {
            REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }
    }
}

// TEST_CASE("Inplace")
// {
//     constexpr uint32_t N = 4;
//     constexpr uint32_t kBlockSize = 8;
//     auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Householder);

//     std::vector<float> input(N * kBlockSize, 0.f);
//     // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
//     for (auto i = 0u; i < N; ++i)
//     {
//         input[i * kBlockSize + i] = 1.f;
//     }

//     sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);

//     mix_mat.Process(input_buffer, input_buffer);

//     // clang-format off
//     constexpr std::array<float, N * kBlockSize> expected = {
//          0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
//     // clang-format on

//     for (auto i = 0u; i < input.size(); i += N)
//     {
//         REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(input[i], std::numeric_limits<float>::epsilon()));
//     }
// }

TEST_CASE("Hadamard_Block")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (auto i = 0u; i < N; ++i)
    {
        input[(i * kBlockSize) + i] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    mix_mat.Process(input_buffer, output_buffer);

    // clang-format off
    constexpr std::array<float, N * kBlockSize> expected = {
        0.5000,  0.5000,  0.5000,  0.5000,  0, 0, 0, 0,
        0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (auto i = 0u; i < input.size(); i += N)
    {
        REQUIRE_THAT(expected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("MatrixAssignment")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat(N);

    std::array<float, N * N> matrix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    mix_mat.SetMatrix(matrix);

    std::array<float, N * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, N * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    mix_mat.Process(input_buffer, output_buffer);
}

TEST_CASE("RandomMatrix")
{
    constexpr uint32_t N = 6;
    // clang-format off
    constexpr std::array<float, N * N> kRandomMatrix = {
    0.4775,    0.0774,    0.5334,   -0.6286,    0.2938,    0.0014,
    0.1239,    0.4800,    0.1981,   -0.0300,   -0.7498,   -0.3897,
    0.7826,    0.2378,   -0.4937,    0.2650,    0.1288,    0.0226,
    0.0785,   -0.5241,   -0.4885,   -0.5273,   -0.2291,   -0.3873,
    0.1624,   -0.1916,   -0.0102,   -0.1466,   -0.5125,    0.8079,
    0.3342,   -0.6291,    0.4402,    0.4839,   -0.1404,   -0.2120,
    };
    // clang-format on
    sfFDN::ScalarFeedbackMatrix mix_mat(N);
    mix_mat.SetMatrix(kRandomMatrix);

    std::array<float, N> input = {1, 2, 3, 4, 5, 6};
    std::array<float, N> output = {0.f};

    sfFDN::AudioBuffer input_buffer(1, N, input);
    sfFDN::AudioBuffer output_buffer(1, N, output);

    mix_mat.Process(input_buffer, output_buffer);

    for (auto val : output)
    {
        std::cout << val << " ";
    }
}

TEST_CASE("DelayMatrix")
{
#ifndef __cpp_lib_mdspan
    SKIP();
#endif
    constexpr uint32_t N = 4;
    constexpr uint32_t delays[] = {11, 11, 2, 6, 10, 14, 17, 8, 2, 6, 19, 5, 10, 19, 1, 13};
    sfFDN::ScalarFeedbackMatrix mixing_matrix = sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard);
    sfFDN::DelayMatrix delay_matrix(4, delays, mixing_matrix);

    constexpr uint32_t kBlockSize = 32;
    std::array<float, N * kBlockSize> input = {0.f};
    std::array<float, N * kBlockSize> output = {0.f};

    for (auto i = 0u; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f; // Set the first sample of each channel to 1
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);
    delay_matrix.Process(input_buffer, output_buffer);

    const std::array<float, kBlockSize> expected_output_ch1 = {0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.5, 0, 0, 0, 0,
                                                               0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch2 = {
        0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch3 = {0, -0.5, 0.5, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0, 0.5,  0,   -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::array<float, kBlockSize> expected_output_ch4 = {
        0, 0, 0, 0, 0, -0.5, 0.5, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (auto i = 0u; i < output_buffer.SampleCount(); ++i)
    {
        REQUIRE_THAT(output_buffer.GetChannelSpan(0)[i],
                     Catch::Matchers::WithinAbs(expected_output_ch1[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(1)[i],
                     Catch::Matchers::WithinAbs(expected_output_ch2[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(2)[i],
                     Catch::Matchers::WithinAbs(expected_output_ch3[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(3)[i],
                     Catch::Matchers::WithinAbs(expected_output_ch4[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("FilterFeedbackMatrix")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t K = 1;

    std::vector<sfFDN::ScalarFeedbackMatrix> mixing_matrices;
    mixing_matrices.reserve(K);
    for (uint32_t i = 0; i < K; ++i)
    {
        mixing_matrices.push_back(sfFDN::ScalarFeedbackMatrix(N, sfFDN::ScalarMatrixType::Hadamard));
    }

    // sfFDN::FilterFeedbackMatrix ffm(N);
    // ffm.ConstructMatrix(delays, mixing_matrices);

    auto ffm = CreateFFM(N, K, 3);

    constexpr uint32_t kBlockSize = 64;
    std::array<float, N * kBlockSize> input = {0.f};
    // input[0] = 1.f;

    for (uint32_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::array<float, N * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    ffm->Process(input_buffer, output_buffer);

    // for (auto i = 0u; i < kBlockSize; ++i)
    // {
    //     std::print("{} \t", i + 1);
    //     for (auto j = 0u; j < N; ++j)
    //     {
    //         std::print("{} \t", output_buffer.GetChannelSpan(j)[i]);
    //     }
    //     std::print("\n");
    // }
}