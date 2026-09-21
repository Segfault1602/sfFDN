// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "allocation_counter.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/time_varying_kronecker_feedback_matrix.h"

#include <array>
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

sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions MakeOptions()
{
    return {
        .matrix =
            {
                .matrix_size = 8U,
                .angles = {0.64350110879328437F, -0.9F, 0.39479111969976149F},
                .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection,
                                 sfFDN::KroneckerKernelType::Rotation},
            },
        .time_varying_config = {{.frequency = 0.25F, .amplitude = 0.3F, .initial_phase = 0.96816901138162093F},
                                {.frequency = 7000.0F / 48000.0F,
                                 .amplitude = 0.0F,
                                 .initial_phase = 0.41666666666666674F},
                                {.frequency = 0.0F, .amplitude = -0.2F, .initial_phase = 0.58333333333333326F}},
    };
}

} // namespace

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix matches pinned pyFDN sine modulation",
          "[time_varying_kronecker_feedback_matrix]")
{
    sfFDN::TimeVaryingKroneckerFeedbackMatrix matrix(MakeOptions());
    constexpr std::array source = {1.0F, -2.0F, 3.5F, -4.5F, 5.25F, -6.75F, 7.125F, -8.875F};
    std::vector<float> input(16);
    for (size_t channel = 0; channel < source.size(); ++channel)
    {
        input[channel * 2U] = source[channel];
        input[(channel * 2U) + 1U] = source[channel];
    }
    std::vector<float> output(16, 0.0F);
    sfFDN::AudioBuffer input_buffer(2U, 8U, input);
    sfFDN::AudioBuffer output_buffer(2U, 8U, output);
    matrix.Process(input_buffer, output_buffer);

    constexpr std::array expected = {
        -0.062506428633578359, 0.062047667062990364, -0.1001990420306812, -0.10048377310009404,
        4.6154250881207428,    3.7334928060990875,   -1.880395227218667,  3.3013734144151967,
        -4.3945990590943893,   -3.5888480969992749,  1.8283522968157508,  -3.1265863730479926,
        -12.414499087096782,   -11.072215249453109,  6.2071819329517783,  -8.3698831892725813,
    };
    RequireNear(output, expected, 4.0e-4);

    std::vector<float> dense(64, 0.0F);
    REQUIRE(matrix.GetMatrix(dense, 1U));
    std::vector<double> queried_output(8, 0.0);
    for (uint32_t row = 0; row < 8U; ++row)
    {
        for (uint32_t column = 0; column < 8U; ++column)
        {
            queried_output[row] += dense[(row * 8U) + column] * source[column];
        }
    }
    constexpr std::array expected_second = {0.062047667062990364, -0.10048377310009404, 3.7334928060990875,
                                            3.3013734144151967,   -3.5888480969992749,  -3.1265863730479926,
                                            -11.072215249453109,  -8.3698831892725813};
    for (size_t index = 0; index < queried_output.size(); ++index)
    {
        REQUIRE_THAT(queried_output[index], Catch::Matchers::WithinAbs(expected_second[index], 4.0e-4));
    }
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix is independent of block partitioning",
          "[time_varying_kronecker_feedback_matrix]")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kSampleCount = 1085U;
    auto input = MakePlanarInput(kOrder, kSampleCount);
    std::vector<float> whole_output(input.size(), 0.0F);
    std::vector<float> split_output(input.size(), 0.0F);
    sfFDN::AudioBuffer input_buffer(kSampleCount, kOrder, input);
    sfFDN::AudioBuffer whole_buffer(kSampleCount, kOrder, whole_output);
    sfFDN::AudioBuffer split_buffer(kSampleCount, kOrder, split_output);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix whole(MakeOptions());
    sfFDN::TimeVaryingKroneckerFeedbackMatrix split(MakeOptions());
    whole.Process(input_buffer, whole_buffer);

    constexpr std::array kBlocks = {1U, 127U, 128U, 37U, 64U, 100U, 129U, 300U, 199U};
    uint32_t offset = 0U;
    for (const uint32_t block_size : kBlocks)
    {
        const auto input_block = input_buffer.Offset(offset, block_size);
        auto output_block = split_buffer.Offset(offset, block_size);
        split.Process(input_block, output_block);
        offset += block_size;
    }
    REQUIRE(offset == kSampleCount);
    REQUIRE(split_output == whole_output);
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix preserves phase through zero depth",
          "[time_varying_kronecker_feedback_matrix]")
{
    auto zero_depth = MakeOptions();
    auto active = zero_depth;
    for (auto& modulation : zero_depth.time_varying_config)
    {
        modulation.amplitude = 0.0F;
    }

    constexpr uint32_t kPrefix = 300U;
    auto prefix = MakePlanarInput(8U, kPrefix);
    std::vector<float> discarded(prefix.size(), 0.0F);
    sfFDN::AudioBuffer prefix_input(kPrefix, 8U, prefix);
    sfFDN::AudioBuffer discarded_buffer(kPrefix, 8U, discarded);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix transitioned(zero_depth);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix reference(active);
    transitioned.Process(prefix_input, discarded_buffer);
    reference.Process(prefix_input, discarded_buffer);
    transitioned.SetTimeVaryingConfig(active.time_varying_config);

    auto suffix = MakePlanarInput(8U, 37U);
    std::vector<float> transitioned_output(suffix.size(), 0.0F);
    std::vector<float> reference_output(suffix.size(), 0.0F);
    sfFDN::AudioBuffer suffix_input(37U, 8U, suffix);
    sfFDN::AudioBuffer transitioned_buffer(37U, 8U, transitioned_output);
    sfFDN::AudioBuffer reference_buffer(37U, 8U, reference_output);
    transitioned.Process(suffix_input, transitioned_buffer);
    reference.Process(suffix_input, reference_buffer);
    REQUIRE(transitioned_output == reference_output);
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix Clear and Clone preserve lifecycle contracts",
          "[time_varying_kronecker_feedback_matrix]")
{
    auto prefix = MakePlanarInput(8U, 19U);
    auto suffix = MakePlanarInput(8U, 31U);
    std::vector<float> discarded(prefix.size(), 0.0F);
    std::vector<float> original_output(suffix.size(), 0.0F);
    std::vector<float> clone_output(suffix.size(), 0.0F);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix matrix(MakeOptions());
    sfFDN::AudioBuffer prefix_input(19U, 8U, prefix);
    sfFDN::AudioBuffer prefix_output(19U, 8U, discarded);
    matrix.Process(prefix_input, prefix_output);
    auto clone = matrix.Clone();
    sfFDN::AudioBuffer suffix_input(31U, 8U, suffix);
    sfFDN::AudioBuffer original_buffer(31U, 8U, original_output);
    sfFDN::AudioBuffer clone_buffer(31U, 8U, clone_output);
    matrix.Process(suffix_input, original_buffer);
    clone->Process(suffix_input, clone_buffer);
    REQUIRE(clone_output == original_output);

    matrix.Clear();
    sfFDN::TimeVaryingKroneckerFeedbackMatrix fresh(MakeOptions());
    std::ranges::fill(original_output, 0.0F);
    std::ranges::fill(clone_output, 0.0F);
    matrix.Process(suffix_input, original_buffer);
    fresh.Process(suffix_input, clone_buffer);
    REQUIRE(original_output == clone_output);
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix setters are transactional", "[time_varying_kronecker_feedback_matrix]")
{
    sfFDN::TimeVaryingKroneckerFeedbackMatrix matrix(MakeOptions());
    std::vector<float> before(64, 0.0F);
    std::vector<float> after(64, 0.0F);
    REQUIRE(matrix.GetMatrix(before, 37U));

    REQUIRE_THROWS_AS(matrix.SetAngles(std::array{0.0F}), std::invalid_argument);
    REQUIRE(matrix.GetMatrix(after, 37U));
    REQUIRE(after == before);

    auto invalid_modulation = MakeOptions().time_varying_config;
    invalid_modulation[1].amplitude = 1.1F;
    REQUIRE_THROWS_AS(matrix.SetTimeVaryingConfig(invalid_modulation), std::invalid_argument);
    std::ranges::fill(after, 0.0F);
    REQUIRE(matrix.GetMatrix(after, 37U));
    REQUIRE(after == before);

    invalid_modulation.pop_back();
    REQUIRE_THROWS_AS(matrix.SetTimeVaryingConfig(invalid_modulation), std::invalid_argument);
    std::ranges::fill(after, 0.0F);
    REQUIRE(matrix.GetMatrix(after, 37U));
    REQUIRE(after == before);
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix processes in place and respects offset views",
          "[time_varying_kronecker_feedback_matrix]")
{
    constexpr uint32_t kOrder = 8U;
    constexpr uint32_t kSamples = 137U;
    constexpr uint32_t kGuard = 5U;
    auto input_data = MakePlanarInput(kOrder, kSamples + (2U * kGuard));
    const auto original_input = input_data;
    auto in_place_data = input_data;
    std::vector<float> output_data(input_data.size(), 99.0F);
    sfFDN::AudioBuffer input_full(kSamples + (2U * kGuard), kOrder, input_data);
    sfFDN::AudioBuffer in_place_full(kSamples + (2U * kGuard), kOrder, in_place_data);
    sfFDN::AudioBuffer output_full(kSamples + (2U * kGuard), kOrder, output_data);
    const auto input = input_full.Offset(kGuard, kSamples);
    auto in_place = in_place_full.Offset(kGuard, kSamples);
    auto output = output_full.Offset(kGuard, kSamples);
    sfFDN::TimeVaryingKroneckerFeedbackMatrix in_place_matrix(MakeOptions());
    sfFDN::TimeVaryingKroneckerFeedbackMatrix out_of_place_matrix(MakeOptions());
    in_place_matrix.Process(in_place, in_place);
    out_of_place_matrix.Process(input, output);

    REQUIRE(input_data == original_input);
    for (uint32_t channel = 0; channel < kOrder; ++channel)
    {
        const auto in_place_channel = in_place_full.GetChannelSpan(channel);
        const auto output_channel = output_full.GetChannelSpan(channel);
        for (uint32_t sample = 0; sample < kSamples; ++sample)
        {
            REQUIRE(in_place_channel[kGuard + sample] == output_channel[kGuard + sample]);
        }
        for (uint32_t sample = 0; sample < kGuard; ++sample)
        {
            REQUIRE(in_place_channel[sample] ==
                    original_input[(static_cast<size_t>(channel) * (kSamples + (2U * kGuard))) + sample]);
            REQUIRE(in_place_channel[kGuard + kSamples + sample] ==
                    original_input[(static_cast<size_t>(channel) * (kSamples + (2U * kGuard))) + kGuard + kSamples +
                                   sample]);
            REQUIRE(output_channel[sample] == 99.0F);
            REQUIRE(output_channel[kGuard + kSamples + sample] == 99.0F);
        }
    }
}

TEST_CASE("TimeVaryingKroneckerFeedbackMatrix validates and does not allocate",
          "[time_varying_kronecker_feedback_matrix]")
{
    auto invalid = MakeOptions();
    invalid.time_varying_config.pop_back();
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingKroneckerFeedbackMatrix(invalid), std::invalid_argument);

    sfFDN::TimeVaryingKroneckerFeedbackMatrix matrix(MakeOptions());
    auto input = MakePlanarInput(8U, 129U);
    std::vector<float> output(input.size(), 0.0F);
    sfFDN::AudioBuffer input_buffer(129U, 8U, input);
    sfFDN::AudioBuffer output_buffer(129U, 8U, output);
    {
        const sfFDNTest::ScopedAllocationCounter allocation_counter;
        matrix.Process(input_buffer, output_buffer);
        matrix.Process(input_buffer, output_buffer);
        REQUIRE(allocation_counter.Count() == 0U);
    }

    std::array<float, 63> wrong_size{};
    wrong_size.fill(7.0F);
    REQUIRE_FALSE(matrix.GetMatrix(wrong_size));
    REQUIRE(std::ranges::all_of(wrong_size, [](float value) { return value == 7.0F; }));

    std::vector<float> empty_storage;
    sfFDN::AudioBuffer empty_input(0U, 8U, empty_storage);
    sfFDN::AudioBuffer empty_output(0U, 8U, empty_storage);
    REQUIRE_NOTHROW(matrix.Process(empty_input, empty_output));
}
