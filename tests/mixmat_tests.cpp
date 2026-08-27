#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <limits>
#include <ranges>

#include "sffdn/audio_buffer.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "matrix_multiplication.h"
#include "test_utils.h"

TEST_CASE("VelvetFFM")
{
    constexpr uint32_t kStageCount = 4;
    constexpr float kSparsity = 3.f;
    constexpr uint32_t kMatSize = 4;
    constexpr float kCascadeGain = 1.f;

    sfFDN::CascadedFeedbackMatrixOptions ffm_info = {.matrix_size = kMatSize,
                                                     .stage_count = kStageCount,
                                                     .sparsity = kSparsity,
                                                     .type = sfFDN::ScalarMatrixType::Random,
                                                     .gain_per_samples = kCascadeGain};

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(ffm_info);
    REQUIRE(ffm != nullptr);

    constexpr uint32_t kBlockSize = 16;
    std::vector<float> input_buffer_data(kMatSize * kBlockSize, 0.f);
    std::vector<float> output_buffer_data(kMatSize * kBlockSize, 0.f);

    // Impulse input
    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input_buffer_data);
    input_buffer.GetChannelSpan(0)[0] = 1.f;

    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output_buffer_data);

    ffm->Process(input_buffer, output_buffer);

    for (auto i = 0u; i < kMatSize; ++i)
    {
        std::cout << "Output Channel " << i << ": ";
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            std::cout << output_buffer.GetChannelSpan(i)[j] << " ";
        }
        std::cout << "\n";
    }
}

TEST_CASE("VariableDiffusionMatrix")
{
    constexpr uint32_t kMatSize = 8;
    auto mat = sfFDN::GenerateMatrix(kMatSize, sfFDN::ScalarMatrixType::VariableDiffusion, 0.f, 1.0f);

    for (auto i = 0u; i < kMatSize; ++i)
    {
        for (auto j = 0u; j < kMatSize; ++j)
        {
            std::cout << mat[i * kMatSize + j] << " ";
        }
        std::cout << "\n";
    }
}

TEST_CASE("IdentityMatrix")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat({kMatSize, sfFDN::ScalarMatrixType::Identity});

    std::array<float, kMatSize * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, kMatSize * kBlockSize> output{};

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);

    for (const auto [in, out] : std::views::zip(input, output))
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

TEST_CASE("ScalarFeedbackMatrix supports aliased processing")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 3;
    constexpr std::array<float, kMatSize * kMatSize> kMatrix = {1.f, 0.f,   0.f, 0.f, 0.5f, 1.f, 0.f,   0.f,
                                                                0.f, 0.25f, 1.f, 0.f, 0.f,  0.f, 0.75f, 1.f};

    sfFDN::ScalarFeedbackMatrix matrix(
        {.matrix_size = kMatSize, .custom_matrix = std::vector<float>(kMatrix.begin(), kMatrix.end())});

    std::array<float, kMatSize * kBlockSize> input = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    std::array<float, kMatSize * kBlockSize> expected{};
    auto in_place = input;

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer expected_buffer(kBlockSize, kMatSize, expected);
    matrix.Process(input_buffer, expected_buffer);

    sfFDN::AudioBuffer in_place_buffer(kBlockSize, kMatSize, in_place);
    matrix.Process(in_place_buffer, in_place_buffer);

    for (const auto [actual, expected_sample] : std::views::zip(in_place, expected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected_sample, 1e-5f));
    }
}

TEST_CASE("Householder")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Householder});

    std::vector<float> input(kMatSize * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (auto i = 0u; i < kMatSize; ++i)
    {
        input[i * kBlockSize + i] = 1.f;
    }

    std::vector<float> output(kMatSize * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);

    // clang-format off
    constexpr std::array<float, kMatSize * kBlockSize> kExpected = {
         0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], 2e-5f));
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

TEST_CASE("FeedbackMatrixHadamard")
{
    SECTION("Hadamard_4")
    {
        constexpr uint32_t kMatSize = 4;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Hadamard});

        std::array<float, kMatSize> input = {1, 2, 3, 4};
        std::array<float, kMatSize> output{};

        sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
        sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, kMatSize> kExpected = {5, -1, -2, 0};

        for (auto i = 0u; i < input.size(); ++i)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], 2e-5f));
        }
    }

    SECTION("Hadamard_8")
    {
        constexpr uint32_t kMatSize = 8;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Hadamard});

        std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, kMatSize> output{};

        sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
        sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, kMatSize> kExpected = {
            12.727922061357855f, -1.414213562373095f, -2.828427124746190f, 0.f, -5.656854249492380f, 0.f, 0.f, 0.f};

        for (auto i = 0u; i < input.size(); ++i)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], 2e-5f));
        }
    }

    SECTION("Hadamard_16")
    {
        constexpr uint32_t kMatSize = 16;
        auto mix_mat = sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Hadamard});

        std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, kMatSize> output{};

        sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
        sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, kMatSize> kExpected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }
    }
}

// TEST_CASE("Inplace")
// {
//     constexpr uint32_t kMatSize = 4;
//     constexpr uint32_t kBlockSize = 8;
//     auto mix_mat = sfFDN::ScalarFeedbackMatrix(kMatSize, sfFDN::ScalarMatrixType::Householder);

//     std::vector<float> input(kMatSize * kBlockSize, 0.f);
//     // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
//     for (auto i = 0u; i < kMatSize; ++i)
//     {
//         input[i * kBlockSize + i] = 1.f;
//     }

//     sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);

//     mix_mat.Process(input_buffer, input_buffer);

//     // clang-format off
//     constexpr std::array<float, kMatSize * kBlockSize> kExpected = {
//          0.5000, -0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
//         -0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
//     // clang-format on

//     for (auto i = 0u; i < input.size(); i += kMatSize)
//     {
//         REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(input[i], std::numeric_limits<float>::epsilon()));
//     }
// }

TEST_CASE("Hadamard_Block")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat = sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Hadamard});

    std::vector<float> input(kMatSize * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (auto i = 0u; i < kMatSize; ++i)
    {
        input[(i * kBlockSize) + i] = 1.f;
    }

    std::vector<float> output(kMatSize * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);

    // clang-format off
    constexpr std::array<float, kMatSize * kBlockSize> kExpected = {
        0.5000,  0.5000,  0.5000,  0.5000,  0, 0, 0, 0,
        0.5000, -0.5000,  0.5000, -0.5000,  0, 0, 0, 0,
        0.5000,  0.5000, -0.5000, -0.5000,  0, 0, 0, 0,
        0.5000, -0.5000, -0.5000,  0.5000,  0, 0, 0, 0};
    // clang-format on

    for (auto i = 0u; i < input.size(); i += kMatSize)
    {
        REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("MatrixAssignment")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat({kMatSize});

    std::array<float, kMatSize * kMatSize> matrix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    mix_mat.SetMatrix(matrix);

    std::array<float, kMatSize * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, kMatSize * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);
}

TEST_CASE("RandomMatrix")
{
    constexpr uint32_t kMatSize = 6;

    sfFDN::ScalarFeedbackMatrix mix_mat({kMatSize, sfFDN::ScalarMatrixType::Random});

    std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6};
    std::array<float, kMatSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

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
    constexpr uint32_t kMatSize = 4;
    constexpr std::array<uint32_t, 16> kDelays = {11, 11, 2, 6, 10, 14, 17, 8, 2, 6, 19, 5, 10, 19, 1, 13};
    sfFDN::ScalarFeedbackMatrix mixing_matrix =
        sfFDN::ScalarFeedbackMatrix({kMatSize, sfFDN::ScalarMatrixType::Hadamard});
    sfFDN::DelayMatrix delay_matrix(4, kDelays, mixing_matrix);

    constexpr uint32_t kBlockSize = 32;
    std::array<float, kMatSize * kBlockSize> input = {0.f};
    std::array<float, kMatSize * kBlockSize> output = {0.f};

    for (auto i = 0u; i < kMatSize; ++i)
    {
        input[i * kBlockSize] = 1.f; // Set the first sample of each channel to 1
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);
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
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kStageCount = 1;

    auto ffm = CreateFFM(kMatSize, kStageCount, 3);

    constexpr uint32_t kBlockSize = 64;
    std::array<float, kMatSize * kBlockSize> input = {0.f};
    // input[0] = 1.f;

    for (uint32_t i = 0; i < kMatSize; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::array<float, kMatSize * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    ffm->Process(input_buffer, output_buffer);

    // for (auto i = 0u; i < kBlockSize; ++i)
    // {
    //     std::print("{} \t", i + 1);
    //     for (auto j = 0u; j < kMatSize; ++j)
    //     {
    //         std::print("{} \t", output_buffer.GetChannelSpan(j)[i]);
    //     }
    //     std::print("\n");
    // }
}

TEST_CASE("Structured feedback matrices match dense processing without allocations")
{
    constexpr std::array kOrders = {8u, 16u};
    constexpr std::array kBlockSizes = {64u, 128u};
    constexpr std::array kTypes = {sfFDN::ScalarMatrixType::Hadamard, sfFDN::ScalarMatrixType::Householder};

    for (const auto type : kTypes)
    {
        for (const auto order : kOrders)
        {
            for (const auto block_size : kBlockSizes)
            {
                std::vector<float> input(order * block_size);
                for (auto i = 0u; i < input.size(); ++i)
                {
                    input[i] = static_cast<float>(static_cast<int>((i * 37u) % 101u) - 50) / 50.f;
                }

                const auto matrix_data = sfFDN::GenerateMatrix(order, type);
                sfFDN::ScalarFeedbackMatrix structured({.matrix_size = order, .type = type});
                sfFDN::ScalarFeedbackMatrix dense({.matrix_size = order, .type = type, .custom_matrix = matrix_data});

                std::vector<float> expected(input.size());
                std::vector<float> actual(input.size());
                auto aliased = input;
                sfFDN::AudioBuffer input_buffer(block_size, order, input);
                sfFDN::AudioBuffer expected_buffer(block_size, order, expected);
                sfFDN::AudioBuffer actual_buffer(block_size, order, actual);
                sfFDN::AudioBuffer aliased_buffer(block_size, order, aliased);

                dense.Process(input_buffer, expected_buffer);

                std::size_t allocations = 0;
                {
                    sfFDNTest::ScopedAllocationCounter allocation_counter;
                    structured.Process(input_buffer, actual_buffer);
                    structured.Process(aliased_buffer, aliased_buffer);
                    allocations = allocation_counter.Count();
                }

                INFO("type=" << static_cast<int>(type) << " order=" << order << " block=" << block_size);
                REQUIRE(allocations == 0);
                for (auto i = 0u; i < actual.size(); ++i)
                {
                    REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 2e-5f));
                    REQUIRE_THAT(aliased[i], Catch::Matchers::WithinAbs(expected[i], 2e-5f));
                }
            }
        }
    }
}

TEST_CASE("SetMatrix disables structured processing")
{
    constexpr uint32_t kOrder = 8;
    constexpr uint32_t kBlockSize = 3;
    const auto matrix_data = sfFDN::GenerateMatrix(kOrder, sfFDN::ScalarMatrixType::Random);
    sfFDN::ScalarFeedbackMatrix updated({.matrix_size = kOrder, .type = sfFDN::ScalarMatrixType::Hadamard});
    sfFDN::ScalarFeedbackMatrix dense(
        {.matrix_size = kOrder, .type = sfFDN::ScalarMatrixType::Random, .custom_matrix = matrix_data});
    REQUIRE(updated.SetMatrix(matrix_data));

    std::array<float, kOrder * kBlockSize> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i + 1);
    }
    std::array<float, kOrder * kBlockSize> expected{};
    std::array<float, kOrder * kBlockSize> actual{};
    sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer expected_buffer(kBlockSize, kOrder, expected);
    sfFDN::AudioBuffer actual_buffer(kBlockSize, kOrder, actual);
    dense.Process(input_buffer, expected_buffer);
    updated.Process(input_buffer, actual_buffer);

    for (auto i = 0u; i < actual.size(); ++i)
    {
        REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 2e-5f));
    }
}

TEST_CASE("FilterFeedbackMatrix uses structured stage-zero processing")
{
    constexpr uint32_t kOrder = 16;
    constexpr uint32_t kBlockSize = 64;
    constexpr std::array kTypes = {sfFDN::ScalarMatrixType::Hadamard, sfFDN::ScalarMatrixType::Householder};

    for (const auto type : kTypes)
    {
        sfFDN::FilterFeedbackMatrix ffm({
            .matrix_size = kOrder,
            .stage_count = 0,
            .sparsity = 1.f,
            .type = type,
            .gain_per_samples = 1.f,
        });
        const auto matrix_data = sfFDN::GenerateMatrix(kOrder, type);
        sfFDN::ScalarFeedbackMatrix dense({.matrix_size = kOrder, .type = type, .custom_matrix = matrix_data});

        std::array<float, kOrder * kBlockSize> input{};
        for (auto i = 0u; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(static_cast<int>((i * 41u) % 113u) - 56) / 56.f;
        }
        std::array<float, kOrder * kBlockSize> expected{};
        auto actual = input;
        sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
        sfFDN::AudioBuffer expected_buffer(kBlockSize, kOrder, expected);
        sfFDN::AudioBuffer actual_buffer(kBlockSize, kOrder, actual);
        dense.Process(input_buffer, expected_buffer);

        std::size_t allocations = 0;
        {
            sfFDNTest::ScopedAllocationCounter allocation_counter;
            ffm.Process(actual_buffer, actual_buffer);
            allocations = allocation_counter.Count();
        }

        REQUIRE(allocations == 0);
        for (auto i = 0u; i < actual.size(); ++i)
        {
            REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 2e-5f));
        }
    }
}