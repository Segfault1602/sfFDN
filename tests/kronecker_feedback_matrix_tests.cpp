// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "allocation_counter.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/kronecker_feedback_matrix.h"

#include <array>
#include <bit>
#include <cmath>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>

namespace
{

void RequireNear(std::span<const float> actual, std::span<const double> expected, double tolerance)
{
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
    {
        REQUIRE_THAT(static_cast<double>(actual[index]), Catch::Matchers::WithinAbs(expected[index], tolerance));
    }
}

std::vector<double> DenseKroneckerMatrix(std::span<const double> angles,
                                         std::span<const sfFDN::KroneckerKernelType> kinds)
{
    std::vector<double> matrix{1.0};
    size_t order = 1U;
    for (size_t stage = 0; stage < angles.size(); ++stage)
    {
        const double sine = std::sin(angles[stage]);
        const double cosine = std::cos(angles[stage]);
        const std::array kernel = kinds[stage] == sfFDN::KroneckerKernelType::Rotation
                                      ? std::array{cosine, -sine, sine, cosine}
                                      : std::array{cosine, sine, sine, -cosine};
        const size_t new_order = 2U * order;
        std::vector<double> expanded(new_order * new_order, 0.0);
        for (size_t kernel_row = 0; kernel_row < 2U; ++kernel_row)
        {
            for (size_t kernel_column = 0; kernel_column < 2U; ++kernel_column)
            {
                for (size_t row = 0; row < order; ++row)
                {
                    for (size_t column = 0; column < order; ++column)
                    {
                        expanded[((kernel_row * order + row) * new_order) + (kernel_column * order + column)] =
                            kernel[(kernel_row * 2U) + kernel_column] * matrix[(row * order) + column];
                    }
                }
            }
        }
        matrix = std::move(expanded);
        order = new_order;
    }
    return matrix;
}

std::vector<float> MakePlanarInput(uint32_t order, uint32_t sample_count)
{
    std::vector<float> input(static_cast<size_t>(order) * sample_count);
    for (uint32_t channel = 0; channel < order; ++channel)
    {
        for (uint32_t sample = 0; sample < sample_count; ++sample)
        {
            input[(static_cast<size_t>(channel) * sample_count) + sample] =
                static_cast<float>((channel + 1U) * (sample + 3U) % 19U) / 9.0F - 1.0F;
        }
    }
    return input;
}

} // namespace

TEST_CASE("KroneckerFeedbackMatrix matches the asymmetric static reference", "[kronecker_feedback_matrix]")
{
    const std::array angles = {static_cast<float>(std::atan2(3.0, 4.0)), static_cast<float>(std::atan2(5.0, 12.0))};
    const std::array kinds = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection};
    sfFDN::KroneckerFeedbackMatrix matrix(
        {.matrix_size = 4, .angles = {angles.begin(), angles.end()}, .kernel_types = {kinds.begin(), kinds.end()}});

    std::vector<float> input{1.0F, 2.0F, 4.0F, 8.0F};
    std::vector<float> output(4, 0.0F);
    sfFDN::AudioBuffer input_buffer(1U, 4U, input);
    sfFDN::AudioBuffer output_buffer(1U, 4U, output);
    matrix.Process(input_buffer, output_buffer);

    constexpr std::array expected = {-64.0 / 65.0, 352.0 / 65.0, 86.0 / 65.0, -473.0 / 65.0};
    RequireNear(output, expected, 2.0e-5);

    std::vector<float> dense(16, 0.0F);
    REQUIRE(matrix.GetMatrix(dense));
    constexpr std::array expected_matrix = {
        48.0 / 65.0, -36.0 / 65.0, 20.0 / 65.0,  -15.0 / 65.0, 36.0 / 65.0, 48.0 / 65.0, 15.0 / 65.0,  20.0 / 65.0,
        20.0 / 65.0, -15.0 / 65.0, -48.0 / 65.0, 36.0 / 65.0,  15.0 / 65.0, 20.0 / 65.0, -36.0 / 65.0, -48.0 / 65.0};
    RequireNear(dense, expected_matrix, 2.0e-5);
}

TEST_CASE("KroneckerFeedbackMatrix validates and wraps angles", "[kronecker_feedback_matrix]")
{
    REQUIRE_THROWS_AS(sfFDN::KroneckerFeedbackMatrix({.matrix_size = 3, .angles = {}, .kernel_types = {}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::KroneckerFeedbackMatrix({.matrix_size = 4, .angles = {0.0F}, .kernel_types = {}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::KroneckerFeedbackMatrix(
                          {.matrix_size = 4,
                           .angles = {},
                           .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Count}}),
                      std::invalid_argument);

    sfFDN::KroneckerFeedbackMatrix canonical({.matrix_size = 4, .angles = {0.2F, -0.7F}, .kernel_types = {}});
    sfFDN::KroneckerFeedbackMatrix wrapped(
        {.matrix_size = 4,
         .angles = {0.2F + (2.0F * std::numbers::pi_v<float>), -0.7F - (4.0F * std::numbers::pi_v<float>)},
         .kernel_types = {}});
    std::vector<float> canonical_matrix(16);
    std::vector<float> wrapped_matrix(16);
    REQUIRE(canonical.GetMatrix(canonical_matrix));
    REQUIRE(wrapped.GetMatrix(wrapped_matrix));
    for (size_t index = 0; index < canonical_matrix.size(); ++index)
    {
        REQUIRE_THAT(wrapped_matrix[index], Catch::Matchers::WithinAbs(canonical_matrix[index], 2.0e-6F));
    }
    REQUIRE_THROWS_AS(canonical.SetAngles(std::array{0.0F}), std::invalid_argument);

    const std::array updated_angles = {-0.4F, 0.9F};
    {
        const sfFDNTest::ScopedAllocationCounter allocation_counter;
        canonical.SetAngles(updated_angles);
        REQUIRE(allocation_counter.Count() == 0U);
    }
    std::vector<float> updated_matrix(16);
    REQUIRE(canonical.GetMatrix(updated_matrix));
    REQUIRE(updated_matrix != canonical_matrix);
}

TEST_CASE("KroneckerFeedbackMatrix processes in place without allocation", "[kronecker_feedback_matrix]")
{
    const sfFDN::KroneckerFeedbackMatrixOptions options{.matrix_size = 8, .angles = {}, .kernel_types = {}};
    auto source = MakePlanarInput(8U, 129U);
    auto in_place = source;
    std::vector<float> out_of_place(source.size(), 0.0F);
    sfFDN::AudioBuffer source_buffer(129U, 8U, source);
    sfFDN::AudioBuffer in_place_buffer(129U, 8U, in_place);
    sfFDN::AudioBuffer out_of_place_buffer(129U, 8U, out_of_place);
    sfFDN::KroneckerFeedbackMatrix in_place_matrix(options);
    sfFDN::KroneckerFeedbackMatrix out_of_place_matrix(options);
    in_place_matrix.Process(in_place_buffer, in_place_buffer);
    out_of_place_matrix.Process(source_buffer, out_of_place_buffer);
    REQUIRE(in_place == out_of_place);

    const sfFDNTest::ScopedAllocationCounter allocation_counter;
    in_place_matrix.Process(in_place_buffer, in_place_buffer);
    REQUIRE(allocation_counter.Count() == 0);
}

TEST_CASE("KroneckerFeedbackMatrix matches an independent dense Kronecker oracle", "[kronecker_feedback_matrix]")
{
    for (const uint32_t order : {2U, 4U, 8U, 16U, 32U, 64U})
    {
        const uint32_t stages = std::bit_width(order) - 1U;
        std::vector<float> angles(stages);
        std::vector<double> reference_angles(stages);
        std::vector<sfFDN::KroneckerKernelType> kinds(stages);
        for (uint32_t stage = 0; stage < stages; ++stage)
        {
            angles[stage] = static_cast<float>(0.31 * static_cast<double>(stage + 1U) - 0.47);
            reference_angles[stage] = static_cast<double>(angles[stage]);
            kinds[stage] =
                stage % 2U == 0U ? sfFDN::KroneckerKernelType::Rotation : sfFDN::KroneckerKernelType::Reflection;
        }
        sfFDN::KroneckerFeedbackMatrix matrix({.matrix_size = order, .angles = angles, .kernel_types = kinds});
        std::vector<float> input(order);
        for (uint32_t channel = 0; channel < order; ++channel)
        {
            input[channel] = static_cast<float>((channel * 7U) % 13U) / 6.0F - 1.0F;
        }
        const auto dense = DenseKroneckerMatrix(reference_angles, kinds);
        std::vector<double> expected(order, 0.0);
        for (uint32_t row = 0; row < order; ++row)
        {
            for (uint32_t column = 0; column < order; ++column)
            {
                expected[row] += dense[(static_cast<size_t>(row) * order) + column] * input[column];
            }
        }
        std::vector<float> output(order, 0.0F);
        sfFDN::AudioBuffer input_buffer(1U, order, input);
        sfFDN::AudioBuffer output_buffer(1U, order, output);
        matrix.Process(input_buffer, output_buffer);
        RequireNear(output, expected, 8.0e-5);
    }
}

TEST_CASE("KroneckerFeedbackMatrix angle offset hook matches the dense oracle", "[kronecker_feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kSamples = 5U;
    constexpr std::array kKinds = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection};
    constexpr std::array kBaseAngles = {0.2F, -0.7F};
    sfFDN::KroneckerFeedbackMatrix matrix({.matrix_size = kOrder,
                                           .angles = {kBaseAngles.begin(), kBaseAngles.end()},
                                           .kernel_types = {kKinds.begin(), kKinds.end()}});
    auto input = MakePlanarInput(kOrder, kSamples);
    std::vector<float> output(input.size(), 0.0F);
    std::vector<float> offsets = {
        0.0F, 0.1F, -0.2F, 0.3F, -0.4F, 0.25F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    sfFDN::AudioBuffer input_buffer(kSamples, kOrder, input);
    sfFDN::AudioBuffer output_buffer(kSamples, kOrder, output);
    matrix.ProcessWithAngleOffsets(input_buffer, output_buffer, offsets, 1U);

    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        const std::array angles = {static_cast<double>(kBaseAngles[0] + offsets[sample]),
                                   static_cast<double>(kBaseAngles[1] + offsets[kSamples])};
        const auto dense = DenseKroneckerMatrix(angles, kKinds);
        for (uint32_t row = 0; row < kOrder; ++row)
        {
            double expected = 0.0;
            for (uint32_t column = 0; column < kOrder; ++column)
            {
                expected += dense[(row * kOrder) + column] * input[(column * kSamples) + sample];
            }
            REQUIRE_THAT(output[(row * kSamples) + sample], Catch::Matchers::WithinAbs(expected, 4.0e-5));
        }
    }

    std::vector<float> static_output(input.size(), 0.0F);
    std::vector<float> hooked_output(input.size(), 0.0F);
    sfFDN::AudioBuffer static_buffer(kSamples, kOrder, static_output);
    sfFDN::AudioBuffer hooked_buffer(kSamples, kOrder, hooked_output);
    matrix.Process(input_buffer, static_buffer);
    std::ranges::fill(offsets, 0.0F);
    matrix.ProcessWithAngleOffsets(input_buffer, hooked_buffer, offsets, 0U);
    REQUIRE(static_output == hooked_output);
}

TEST_CASE("KroneckerFeedbackMatrix Clear and Clone preserve static configuration", "[kronecker_feedback_matrix]")
{
    sfFDN::KroneckerFeedbackMatrix matrix(
        {.matrix_size = 4,
         .angles = {0.2F, -0.7F},
         .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection}});
    std::vector<float> before(16);
    std::vector<float> after(16);
    REQUIRE(matrix.GetMatrix(before));
    matrix.Clear();
    REQUIRE(matrix.GetMatrix(after));
    REQUIRE(before == after);
    auto clone = matrix.Clone();
    auto* typed_clone = dynamic_cast<sfFDN::KroneckerFeedbackMatrix*>(clone.get());
    REQUIRE(typed_clone != nullptr);
    std::ranges::fill(after, 0.0F);
    REQUIRE(typed_clone->GetMatrix(after));
    REQUIRE(before == after);
}

TEST_CASE("KroneckerFeedbackMatrix respects offset views and preserves out of place input",
          "[kronecker_feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kSamples = 37U;
    constexpr uint32_t kGuard = 5U;
    sfFDN::KroneckerFeedbackMatrix matrix(
        {.matrix_size = kOrder,
         .angles = {0.3F, -0.8F},
         .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection}});
    auto input_data = MakePlanarInput(kOrder, kSamples + (2U * kGuard));
    const auto original_input = input_data;
    std::vector<float> output_data(input_data.size(), 99.0F);
    sfFDN::AudioBuffer input_full(kSamples + (2U * kGuard), kOrder, input_data);
    sfFDN::AudioBuffer output_full(kSamples + (2U * kGuard), kOrder, output_data);
    const auto input = input_full.Offset(kGuard, kSamples);
    auto output = output_full.Offset(kGuard, kSamples);
    matrix.Process(input, output);
    REQUIRE(input_data == original_input);
    for (uint32_t channel = 0; channel < kOrder; ++channel)
    {
        const auto channel_data = output_full.GetChannelSpan(channel);
        for (uint32_t sample = 0; sample < kGuard; ++sample)
        {
            REQUIRE(channel_data[sample] == 99.0F);
            REQUIRE(channel_data[kGuard + kSamples + sample] == 99.0F);
        }
    }
}
