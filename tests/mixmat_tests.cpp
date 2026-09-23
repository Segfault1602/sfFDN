#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "sffdn/audio_buffer.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/matrix_data.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "matrix_multiplication.h"
#include "test_utils.h"

namespace
{
std::vector<float> DenseReference(std::span<const float> matrix, uint32_t order, uint32_t block_size,
                                  std::span<const float> input)
{
    std::vector<float> output(input.size(), 0.f);
    for (uint32_t destination = 0; destination < order; ++destination)
    {
        for (uint32_t sample = 0; sample < block_size; ++sample)
        {
            for (uint32_t source = 0; source < order; ++source)
            {
                output[(destination * block_size) + sample] +=
                    matrix[(destination * order) + source] * input[(source * block_size) + sample];
            }
        }
    }
    return output;
}

void RequireFiniteOrthogonal(std::span<const float> matrix, uint32_t order)
{
    for (const float coefficient : matrix)
    {
        REQUIRE(std::isfinite(coefficient));
    }
    for (uint32_t row = 0; row < order; ++row)
    {
        for (uint32_t other_row = 0; other_row < order; ++other_row)
        {
            float dot = 0.f;
            for (uint32_t column = 0; column < order; ++column)
            {
                dot += matrix[(row * order) + column] * matrix[(other_row * order) + column];
            }
            REQUIRE_THAT(dot, Catch::Matchers::WithinAbs(row == other_row ? 1.f : 0.f, 1e-5f));
        }
    }
}

void RequireNear(std::span<const float> actual, std::span<const float> expected, float tolerance = 2e-5f)
{
    REQUIRE(actual.size() == expected.size());
    for (const auto [actual_value, expected_value] : std::views::zip(actual, expected))
    {
        REQUIRE_THAT(actual_value, Catch::Matchers::WithinAbs(expected_value, tolerance));
    }
}

std::vector<float> DenseTestMatrix(uint32_t order)
{
    std::vector<float> matrix(static_cast<size_t>(order) * order);
    for (uint32_t row = 0; row < order; ++row)
    {
        for (uint32_t column = 0; column < order; ++column)
        {
            matrix[(row * order) + column] = static_cast<float>((row * order) + column + 1);
        }
    }
    return matrix;
}

void FillLogicalInput(sfFDN::AudioBuffer& buffer)
{
    for (uint32_t channel = 0; channel < buffer.ChannelCount(); ++channel)
    {
        const auto channel_data = buffer.GetChannelSpan(channel);
        for (uint32_t sample = 0; sample < buffer.SampleCount(); ++sample)
        {
            channel_data[sample] = static_cast<float>(static_cast<int>((channel * 31U) + (sample * 7U)) - 19) / 16.f;
        }
    }
}

std::vector<float> DenseReference(const std::span<const float> matrix, const sfFDN::AudioBuffer& input)
{
    const uint32_t order = input.ChannelCount();
    const uint32_t block_size = input.SampleCount();
    std::vector<float> packed_input(static_cast<size_t>(order) * block_size);
    for (uint32_t channel = 0; channel < order; ++channel)
    {
        const auto channel_input = input.GetChannelSpan(channel);
        std::ranges::copy(channel_input, packed_input.begin() + (static_cast<size_t>(channel) * block_size));
    }
    return DenseReference(matrix, order, block_size, packed_input);
}

void RequireLogicalOutput(const sfFDN::AudioBuffer& output, const std::span<const float> expected)
{
    const uint32_t block_size = output.SampleCount();
    for (uint32_t channel = 0; channel < output.ChannelCount(); ++channel)
    {
        const auto channel_output = output.GetChannelSpan(channel);
        const auto channel_expected = expected.subspan(static_cast<size_t>(channel) * block_size, block_size);
        RequireNear(channel_output, channel_expected);
    }
}

void RequireStrideSentinels(const std::span<const float> storage, uint32_t stride, uint32_t channels,
                            uint32_t logical_offset, uint32_t logical_size, float sentinel)
{
    for (uint32_t channel = 0; channel < channels; ++channel)
    {
        const auto channel_storage = storage.subspan(static_cast<size_t>(channel) * stride, stride);
        for (uint32_t sample = 0; sample < stride; ++sample)
        {
            if (sample < logical_offset || sample >= logical_offset + logical_size)
            {
                REQUIRE(channel_storage[sample] == sentinel);
            }
        }
    }
}

std::vector<float> RenderCascade(sfFDN::FilterFeedbackMatrix& matrix, uint32_t block_size, uint32_t block_count)
{
    const uint32_t order = matrix.InputChannelCount();
    std::vector<float> input(static_cast<size_t>(order) * block_size * block_count, 0.f);
    std::vector<float> output(input.size(), 0.f);
    input[0] = 1.f;

    const size_t block_samples = static_cast<size_t>(order) * block_size;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const auto offset = static_cast<size_t>(block) * block_samples;
        sfFDN::AudioBuffer input_buffer(block_size, order, std::span(input).subspan(offset, block_samples));
        sfFDN::AudioBuffer output_buffer(block_size, order, std::span(output).subspan(offset, block_samples));
        matrix.Process(input_buffer, output_buffer);
    }

    return output;
}
} // namespace

TEST_CASE("FilterFeedbackMatrix produces a nonzero response after Clear", "[feedback_matrix]")
{
    constexpr uint32_t kStageCount = 4;
    constexpr float kSparsity = 3.f;
    constexpr uint32_t kMatSize = 4;
    constexpr float kCascadeGain = 1.f;

    sfFDN::CascadedFeedbackMatrixOptions ffm_info = {.matrix_size = kMatSize,
                                                     .stage_count = kStageCount,
                                                     .sparsity = kSparsity,
                                                     .generator = sfFDN::ScalarMatrixType::Random,
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

    float energy = 0.f;
    for (const float sample : output_buffer_data)
    {
        REQUIRE(std::isfinite(sample));
        energy += sample * sample;
    }
    REQUIRE(energy > 0.f);

    ffm->Clear();
    std::ranges::fill(output_buffer_data, 0.f);
    ffm->Process(input_buffer, output_buffer);
    REQUIRE(std::ranges::any_of(output_buffer_data, [](float sample) { return sample != 0.f; }));
}

TEST_CASE("GenerateMatrix creates an orthogonal VariableDiffusion matrix", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 2;
    const auto mat = sfFDN::GenerateMatrix(kMatSize, sfFDN::VariableDiffusionOptions{.diffusion = 0.5f}, 0);
    const float theta = std::numbers::pi_v<float> / 8.f;
    const std::array<float, 4> expected = {std::cos(theta), std::sin(theta), -std::sin(theta), std::cos(theta)};

    REQUIRE(mat.size() == expected.size());
    for (const auto [actual, reference] : std::views::zip(mat, expected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-6f));
    }
    RequireFiniteOrthogonal(mat, kMatSize);
}

TEST_CASE("GetMatrixType identifies both generator alternatives without throwing", "[feedback_matrix]")
{
    const sfFDN::MatrixGeneratorOptions scalar = sfFDN::ScalarMatrixType::Householder;
    const sfFDN::MatrixGeneratorOptions diffusion = sfFDN::VariableDiffusionOptions{.diffusion = 0.5F};
    static_assert(noexcept(sfFDN::GetMatrixType(scalar)));
    REQUIRE(sfFDN::GetMatrixType(scalar) == sfFDN::ScalarMatrixType::Householder);
    REQUIRE(sfFDN::GetMatrixType(diffusion) == sfFDN::ScalarMatrixType::VariableDiffusion);
}

TEST_CASE("MatrixData owns square coefficients and preserves value semantics", "[feedback_matrix]")
{
    static_assert(std::is_copy_constructible_v<sfFDN::MatrixData>);
    static_assert(std::is_copy_assignable_v<sfFDN::MatrixData>);
    static_assert(std::is_nothrow_move_constructible_v<sfFDN::MatrixData>);
    static_assert(std::is_nothrow_move_assignable_v<sfFDN::MatrixData>);

    sfFDN::MatrixData empty;
    REQUIRE(empty.Order() == 0U);
    REQUIRE(empty.Values().empty());

    std::vector<float> source = {1.f, 2.f, 3.f, 4.f};
    sfFDN::MatrixData data(2U, source);
    source[0] = -1.f;
    REQUIRE(data.Values()[0] == 1.f);
    REQUIRE(data.Values().size() == 4U);

    data.Values()[1] = -2.f;
    sfFDN::MatrixData copy = data;
    REQUIRE(copy == data);
    data.Values()[1] = 2.f;
    REQUIRE(copy.Values()[1] == -2.f);
    REQUIRE(copy != data);

    sfFDN::MatrixData moved = std::move(copy);
    REQUIRE(moved.Order() == 2U);
    REQUIRE(moved.Values()[1] == -2.f);
    REQUIRE(copy.Order() == 0U);
    REQUIRE(copy.Values().empty());

    sfFDN::MatrixData moved_empty = std::move(empty);
    REQUIRE(moved_empty.Order() == 0U);
    REQUIRE(moved_empty.Values().empty());
    REQUIRE(empty.Order() == 0U);
    REQUIRE(empty.Values().empty());

    moved = moved;
    REQUIRE(moved.Order() == 2U);
    REQUIRE(moved.Values()[1] == -2.f);

    REQUIRE_THROWS_AS(sfFDN::MatrixData(2U, std::vector<float>(3U, 0.f)), std::invalid_argument);
}

TEST_CASE("ScalarFeedbackMatrix source alternatives are distinct and deterministic", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    const sfFDN::ScalarFeedbackMatrixOptions generated = {
        .source = sfFDN::GeneratedMatrixOptions{.matrix_size = kOrder, .generator = sfFDN::ScalarMatrixType::Random}};
    const sfFDN::ScalarFeedbackMatrixOptions explicit_data = {
        .source = sfFDN::MatrixData{kOrder, std::vector<float>(kOrder * kOrder, 0.f)}};
    REQUIRE(generated.MatrixSize() == kOrder);
    REQUIRE(explicit_data.MatrixSize() == kOrder);
    REQUIRE(generated != explicit_data);

    const sfFDN::ScalarFeedbackMatrixOptions generated_zero = {
        .source = sfFDN::GeneratedMatrixOptions{
            .matrix_size = kOrder, .generator = sfFDN::ScalarMatrixType::Random, .rng_seed = 0U}};
    sfFDN::ScalarFeedbackMatrix default_first(generated);
    sfFDN::ScalarFeedbackMatrix default_repeated(generated);
    sfFDN::ScalarFeedbackMatrix zero_first(generated_zero);
    sfFDN::ScalarFeedbackMatrix zero_repeated(generated_zero);
    std::vector<float> default_matrix(kOrder * kOrder);
    std::vector<float> default_repeated_matrix(kOrder * kOrder);
    std::vector<float> zero_matrix(kOrder * kOrder);
    std::vector<float> zero_repeated_matrix(kOrder * kOrder);
    REQUIRE(default_first.GetMatrix(default_matrix));
    REQUIRE(default_repeated.GetMatrix(default_repeated_matrix));
    REQUIRE(zero_first.GetMatrix(zero_matrix));
    REQUIRE(zero_repeated.GetMatrix(zero_repeated_matrix));
    REQUIRE(default_matrix == default_repeated_matrix);
    REQUIRE(zero_matrix == zero_repeated_matrix);
    REQUIRE(default_matrix != zero_matrix);

    std::array<float, kOrder> input = {1.f, -0.5f, 0.25f, -0.75f};
    std::array<float, kOrder> default_output{};
    std::array<float, kOrder> zero_output{};
    const sfFDN::AudioBuffer input_buffer(1U, kOrder, input);
    sfFDN::AudioBuffer default_output_buffer(1U, kOrder, default_output);
    sfFDN::AudioBuffer zero_output_buffer(1U, kOrder, zero_output);
    default_first.Process(input_buffer, default_output_buffer);
    zero_first.Process(input_buffer, zero_output_buffer);
    REQUIRE(default_output != zero_output);
}

TEST_CASE("ScalarFeedbackMatrix preserves samples with an Identity matrix", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat({.source = sfFDN::GeneratedMatrixOptions{
                                             .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Identity}});

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

TEST_CASE("ScalarFeedbackMatrix supports aliased processing", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 3;
    constexpr std::array<float, kMatSize * kMatSize> kMatrix = {1.f, 0.f,   0.f, 0.f, 0.5f, 1.f, 0.f,   0.f,
                                                                0.f, 0.25f, 1.f, 0.f, 0.f,  0.f, 0.75f, 1.f};

    sfFDN::ScalarFeedbackMatrix matrix(
        {.source = sfFDN::MatrixData{kMatSize, std::vector<float>(kMatrix.begin(), kMatrix.end())}});

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

TEST_CASE("ScalarFeedbackMatrix processes zero-offset strided views", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 3;
    constexpr uint32_t kBlockSize = 5;
    constexpr uint32_t kInputStride = 8;
    constexpr uint32_t kOutputStride = 10;
    constexpr float kSentinel = -9876.5f;
    const auto matrix_data = DenseTestMatrix(kOrder);
    sfFDN::ScalarFeedbackMatrix matrix({.source = sfFDN::MatrixData{kOrder, matrix_data}});
    std::vector<float> input_storage(kOrder * kInputStride, kSentinel);
    std::vector<float> output_storage(kOrder * kOutputStride, kSentinel);
    sfFDN::AudioBuffer input_parent(kInputStride, kOrder, input_storage);
    sfFDN::AudioBuffer output_parent(kOutputStride, kOrder, output_storage);
    sfFDN::AudioBuffer input = input_parent.Offset(0, kBlockSize);
    sfFDN::AudioBuffer output = output_parent.Offset(0, kBlockSize);
    FillLogicalInput(input);
    const auto expected = DenseReference(matrix_data, input);

    matrix.Process(input, output);

    RequireLogicalOutput(output, expected);
    RequireStrideSentinels(input_storage, kInputStride, kOrder, 0, kBlockSize, kSentinel);
    RequireStrideSentinels(output_storage, kOutputStride, kOrder, 0, kBlockSize, kSentinel);

    const auto output_before_empty_process = output_storage;
    sfFDN::AudioBuffer empty_input = input_parent.Offset(0, 0);
    sfFDN::AudioBuffer empty_output = output_parent.Offset(0, 0);
    matrix.Process(empty_input, empty_output);
    REQUIRE(output_storage == output_before_empty_process);

    constexpr std::array kOrders = {4U, 16U, 64U};
    constexpr std::array kBlockSizes = {32U, 128U, 256U, 512U};
    for (const uint32_t order : kOrders)
    {
        const auto sweep_matrix_data = DenseTestMatrix(order);
        sfFDN::ScalarFeedbackMatrix sweep_matrix({.source = sfFDN::MatrixData{order, sweep_matrix_data}});
        for (const uint32_t block_size : kBlockSizes)
        {
            std::vector<float> sweep_input(static_cast<size_t>(order) * block_size, 0.25f);
            std::vector<float> sweep_output(sweep_input.size());
            sfFDN::AudioBuffer sweep_input_buffer(block_size, order, sweep_input);
            sfFDN::AudioBuffer sweep_output_buffer(block_size, order, sweep_output);
            sweep_matrix.Process(sweep_input_buffer, sweep_output_buffer);

            size_t allocations = 0;
            {
                sfFDNTest::ScopedAllocationCounter allocation_counter;
                sweep_matrix.Process(sweep_input_buffer, sweep_output_buffer);
                allocations = allocation_counter.Count();
            }

            std::vector<float> sweep_aliased_input(static_cast<size_t>(order) * block_size, 0.25f);
            sfFDN::AudioBuffer sweep_aliased_buffer(block_size, order, sweep_aliased_input);
            sweep_matrix.Process(sweep_aliased_buffer, sweep_aliased_buffer);

            size_t aliased_allocations = 0;
            {
                sfFDNTest::ScopedAllocationCounter allocation_counter;
                sweep_matrix.Process(sweep_aliased_buffer, sweep_aliased_buffer);
                aliased_allocations = allocation_counter.Count();
            }
            INFO("order=" << order << " block=" << block_size);
            REQUIRE(allocations == 0);
            REQUIRE(aliased_allocations == 0);
        }
    }
}

TEST_CASE("ScalarFeedbackMatrix processes nonzero-offset strided views", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 3;
    constexpr uint32_t kBlockSize = 7;
    constexpr uint32_t kInputStride = 13;
    constexpr uint32_t kOutputStride = 15;
    constexpr uint32_t kInputOffset = 2;
    constexpr uint32_t kOutputOffset = 4;
    constexpr float kSentinel = -4321.25f;
    const auto matrix_data = DenseTestMatrix(kOrder);
    sfFDN::ScalarFeedbackMatrix matrix(
        {.source = sfFDN::GeneratedMatrixOptions{.matrix_size = kOrder, .generator = sfFDN::ScalarMatrixType::Random}});
    REQUIRE(matrix.SetMatrix(matrix_data));

    std::vector<float> input_storage(kOrder * kInputStride, kSentinel);
    std::vector<float> output_storage(kOrder * kOutputStride, kSentinel);
    sfFDN::AudioBuffer input_parent(kInputStride, kOrder, input_storage);
    sfFDN::AudioBuffer output_parent(kOutputStride, kOrder, output_storage);
    sfFDN::AudioBuffer input = input_parent.Offset(kInputOffset, kBlockSize);
    sfFDN::AudioBuffer output = output_parent.Offset(kOutputOffset, kBlockSize);
    FillLogicalInput(input);
    const auto expected = DenseReference(matrix_data, input);

    matrix.Process(input, output);

    std::vector<float> returned_matrix(kOrder * kOrder);
    REQUIRE(matrix.GetMatrix(returned_matrix));
    REQUIRE(returned_matrix == matrix_data);
    RequireLogicalOutput(output, expected);
    RequireStrideSentinels(input_storage, kInputStride, kOrder, kInputOffset, kBlockSize, kSentinel);
    RequireStrideSentinels(output_storage, kOutputStride, kOrder, kOutputOffset, kBlockSize, kSentinel);

    std::vector<float> packed_input(static_cast<size_t>(kOrder) * kBlockSize);
    std::vector<float> packed_output(packed_input.size());
    sfFDN::AudioBuffer packed_input_buffer(kBlockSize, kOrder, packed_input);
    sfFDN::AudioBuffer packed_output_buffer(kBlockSize, kOrder, packed_output);
    for (uint32_t channel = 0; channel < kOrder; ++channel)
    {
        std::ranges::copy(input.GetChannelSpan(channel), packed_input_buffer.GetChannelSpan(channel).begin());
    }
    matrix.Process(packed_input_buffer, packed_output_buffer);
    RequireLogicalOutput(output, packed_output);
}

TEST_CASE("ScalarFeedbackMatrix processes exact strided aliases without allocations", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 3;
    constexpr uint32_t kBlockSize = 37;
    constexpr uint32_t kStride = 43;
    constexpr uint32_t kOffset = 3;
    constexpr float kSentinel = -1357.75f;
    const auto matrix_data = DenseTestMatrix(kOrder);
    sfFDN::ScalarFeedbackMatrix matrix({.source = sfFDN::MatrixData{kOrder, matrix_data}});
    std::vector<float> storage(kOrder * kStride, kSentinel);
    sfFDN::AudioBuffer parent(kStride, kOrder, storage);
    sfFDN::AudioBuffer buffer = parent.Offset(kOffset, kBlockSize);
    FillLogicalInput(buffer);
    const auto expected = DenseReference(matrix_data, buffer);

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        matrix.Process(buffer, buffer);
        allocations = allocation_counter.Count();
    }

    REQUIRE(allocations == 0);
    RequireLogicalOutput(buffer, expected);
    RequireStrideSentinels(storage, kStride, kOrder, kOffset, kBlockSize, kSentinel);

    constexpr uint32_t kFilterOrder = 4;
    constexpr uint32_t kFilterBlockSize = 32;
    const sfFDN::CascadedFeedbackMatrixOptions options = {
        .matrix_size = kFilterOrder,
        .stage_count = 2,
        .sparsity = 2.f,
        .generator = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 0.99f,
        .rng_seed = 12345U,
    };
    sfFDN::FilterFeedbackMatrix packed_filter(options);
    sfFDN::FilterFeedbackMatrix strided_filter(options);
    std::vector<float> packed_storage(kFilterOrder * kFilterBlockSize);
    sfFDN::AudioBuffer packed_buffer(kFilterBlockSize, kFilterOrder, packed_storage);
    FillLogicalInput(packed_buffer);
    constexpr uint32_t kFilterStride = 39;
    constexpr uint32_t kFilterOffset = 4;
    std::vector<float> strided_storage(kFilterOrder * kFilterStride, kSentinel);
    sfFDN::AudioBuffer strided_parent(kFilterStride, kFilterOrder, strided_storage);
    sfFDN::AudioBuffer strided_buffer = strided_parent.Offset(kFilterOffset, kFilterBlockSize);
    for (uint32_t channel = 0; channel < kFilterOrder; ++channel)
    {
        std::ranges::copy(packed_buffer.GetChannelSpan(channel), strided_buffer.GetChannelSpan(channel).begin());
    }

    packed_filter.Process(packed_buffer, packed_buffer);
    strided_filter.Process(strided_buffer, strided_buffer);

    for (uint32_t channel = 0; channel < kFilterOrder; ++channel)
    {
        RequireNear(strided_buffer.GetChannelSpan(channel), packed_buffer.GetChannelSpan(channel), 1e-4f);
    }
    RequireStrideSentinels(strided_storage, kFilterStride, kFilterOrder, kFilterOffset, kFilterBlockSize, kSentinel);
}

TEST_CASE("ScalarFeedbackMatrix dense processing matches the reference across tile remainders", "[feedback_matrix]")
{
    // Orders straddle the row-tile width and the packing threshold; block sizes straddle the SIMD width and the frame
    // tile so every vector, remainder, and scalar path runs both disjoint and in place.
    constexpr std::array kOrders = {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 31U, 32U, 33U, 64U};
    constexpr std::array kBlockSizes = {1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 33U, 130U};
    constexpr uint32_t kOffset = 3;
    constexpr uint32_t kPadding = 5;
    constexpr float kSentinel = -2468.5f;

    for (const uint32_t order : kOrders)
    {
        const auto matrix_data = sfFDN::GenerateMatrix(order, sfFDN::ScalarMatrixType::Random, 97U);
        sfFDN::ScalarFeedbackMatrix matrix({.source = sfFDN::MatrixData{order, matrix_data}});
        for (const uint32_t block_size : kBlockSizes)
        {
            CAPTURE(order, block_size);
            const uint32_t stride = kOffset + block_size + kPadding;
            std::vector<float> input_storage(static_cast<size_t>(order) * stride, kSentinel);
            std::vector<float> output_storage(input_storage.size(), kSentinel);
            sfFDN::AudioBuffer input_parent(stride, order, input_storage);
            sfFDN::AudioBuffer output_parent(stride, order, output_storage);
            sfFDN::AudioBuffer input = input_parent.Offset(kOffset, block_size);
            sfFDN::AudioBuffer output = output_parent.Offset(kOffset, block_size);
            for (uint32_t channel = 0; channel < order; ++channel)
            {
                const auto channel_input = input.GetChannelSpan(channel);
                for (uint32_t sample = 0; sample < block_size; ++sample)
                {
                    channel_input[sample] = std::sin(static_cast<float>((channel * 131U) + (sample * 17U) + 1U));
                }
            }
            const auto expected = DenseReference(matrix_data, input);

            size_t allocations = 0;
            {
                sfFDNTest::ScopedAllocationCounter allocation_counter;
                matrix.Process(input, output);
                matrix.Process(input, input);
                allocations = allocation_counter.Count();
            }

            REQUIRE(allocations == 0);
            RequireLogicalOutput(output, expected);
            RequireLogicalOutput(input, expected);
            RequireStrideSentinels(output_storage, stride, order, kOffset, block_size, kSentinel);
            RequireStrideSentinels(input_storage, stride, order, kOffset, block_size, kSentinel);
        }
    }
}

TEST_CASE("ScalarFeedbackMatrix applies a Householder reflection", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat =
        sfFDN::ScalarFeedbackMatrix({.source = sfFDN::GeneratedMatrixOptions{
                                         .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Householder}});

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

TEST_CASE("ScalarFeedbackMatrix applies Hadamard transforms for supported orders", "[feedback_matrix]")
{
    SECTION("Hadamard_4")
    {
        constexpr uint32_t kMatSize = 4;
        auto mix_mat =
            sfFDN::ScalarFeedbackMatrix({.source = sfFDN::GeneratedMatrixOptions{
                                             .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Hadamard}});

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
        auto mix_mat =
            sfFDN::ScalarFeedbackMatrix({.source = sfFDN::GeneratedMatrixOptions{
                                             .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Hadamard}});

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
        auto mix_mat =
            sfFDN::ScalarFeedbackMatrix({.source = sfFDN::GeneratedMatrixOptions{
                                             .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Hadamard}});

        std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, kMatSize> output{};

        sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
        sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

        mix_mat.Process(input_buffer, output_buffer);

        constexpr std::array<float, kMatSize> kExpected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        for (auto i = 0u; i < input.size(); ++i)
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

TEST_CASE("ScalarFeedbackMatrix applies a Hadamard transform across a block", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 8;
    auto mix_mat =
        sfFDN::ScalarFeedbackMatrix({.source = sfFDN::GeneratedMatrixOptions{
                                         .matrix_size = kMatSize, .generator = sfFDN::ScalarMatrixType::Hadamard}});

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

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("ScalarFeedbackMatrix SetMatrix copies assigned coefficients", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 2;
    sfFDN::ScalarFeedbackMatrix mix_mat({.source = sfFDN::GeneratedMatrixOptions{.matrix_size = kMatSize}});

    std::array<float, kMatSize * kMatSize> matrix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    REQUIRE(mix_mat.SetMatrix(matrix));
    matrix.fill(-1.f);

    std::array<float, kMatSize * kBlockSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, kMatSize * kBlockSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);

    constexpr std::array<float, kMatSize * kBlockSize> kExpected = {34.f,  40.f,  98.f,  120.f,
                                                                    162.f, 200.f, 226.f, 280.f};
    for (const auto [actual, expected] : std::views::zip(output, kExpected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 1e-6f));
    }
}

TEST_CASE("GenerateMatrix creates an orthogonal Random matrix", "[feedback_matrix]")
{
    constexpr uint32_t kMatSize = 6;

    const auto matrix = sfFDN::GenerateMatrix(kMatSize, sfFDN::ScalarMatrixType::Random, 1234);
    RequireFiniteOrthogonal(matrix, kMatSize);
    sfFDN::ScalarFeedbackMatrix mix_mat({.source = sfFDN::MatrixData{kMatSize, matrix}});

    std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6};
    std::array<float, kMatSize> output = {0.f};

    sfFDN::AudioBuffer input_buffer(1, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(1, kMatSize, output);

    mix_mat.Process(input_buffer, output_buffer);

    const auto expected = DenseReference(matrix, kMatSize, 1, input);
    for (const auto [actual, reference] : std::views::zip(output, expected))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-6f));
    }
}

TEST_CASE("ScalarFeedbackMatrix forwards nonzero seeds to generated matrices", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kBlockSize = 3U;
    constexpr uint32_t kSeed = 0x1BADB002U;
    constexpr std::array kTypes = {
        sfFDN::ScalarMatrixType::Random,        sfFDN::ScalarMatrixType::RandomHouseholder,
        sfFDN::ScalarMatrixType::Circulant,     sfFDN::ScalarMatrixType::Allpass,
        sfFDN::ScalarMatrixType::NestedAllpass,
    };
    std::array<float, kOrder * kBlockSize> input = {
        0.25f, -0.5f, 0.75f, 1.f, -0.25f, 0.5f, -0.75f, -1.f, 0.125f, -0.375f, 0.625f, -0.875f,
    };

    for (const auto type : kTypes)
    {
        const auto expected_matrix = sfFDN::GenerateMatrix(kOrder, type, kSeed);
        const auto options = sfFDN::ScalarFeedbackMatrixOptions{
            .source = sfFDN::GeneratedMatrixOptions{.matrix_size = kOrder, .generator = type, .rng_seed = kSeed}};
        sfFDN::ScalarFeedbackMatrix matrix(options);
        sfFDN::ScalarFeedbackMatrix repeated(options);
        std::vector<float> actual_matrix(kOrder * kOrder);
        std::vector<float> repeated_matrix(kOrder * kOrder);
        REQUIRE(matrix.GetMatrix(actual_matrix));
        REQUIRE(repeated.GetMatrix(repeated_matrix));
        INFO("type=" << static_cast<int>(type));
        REQUIRE(actual_matrix == expected_matrix);
        REQUIRE(repeated_matrix == expected_matrix);

        std::array<float, kOrder * kBlockSize> output{};
        sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kOrder, output);
        matrix.Process(input_buffer, output_buffer);
        RequireNear(output, DenseReference(expected_matrix, kOrder, kBlockSize, input));
    }

    const auto first = sfFDN::GenerateMatrix(kOrder, sfFDN::ScalarMatrixType::Random, kSeed);
    const auto second = sfFDN::GenerateMatrix(kOrder, sfFDN::ScalarMatrixType::Random, kSeed + 1U);
    REQUIRE(first != second);
}

TEST_CASE("ScalarFeedbackMatrix forwards VariableDiffusion arguments to processing", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kBlockSize = 2U;
    std::array<float, kOrder * kBlockSize> input = {1.f, -0.5f, 0.25f, -0.75f, 0.5f, -1.f, 0.125f, -0.25f};
    const auto default_matrix = sfFDN::GenerateMatrix(kOrder, sfFDN::ScalarMatrixType::VariableDiffusion);

    for (const float arg : {0.25f, 0.75f})
    {
        const auto expected_matrix =
            sfFDN::GenerateMatrix(kOrder, sfFDN::VariableDiffusionOptions{.diffusion = arg}, 0);
        sfFDN::ScalarFeedbackMatrix matrix(
            {.source = sfFDN::GeneratedMatrixOptions{.matrix_size = kOrder,
                                                     .generator = sfFDN::VariableDiffusionOptions{.diffusion = arg}}});
        std::vector<float> actual_matrix(kOrder * kOrder);
        REQUIRE(matrix.GetMatrix(actual_matrix));
        REQUIRE(actual_matrix == expected_matrix);
        REQUIRE(actual_matrix != default_matrix);

        std::array<float, kOrder * kBlockSize> output{};
        sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kOrder, output);
        matrix.Process(input_buffer, output_buffer);
        RequireNear(output, DenseReference(expected_matrix, kOrder, kBlockSize, input));
    }
}

TEST_CASE("ScalarFeedbackMatrix applies explicit matrix data unchanged", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 3U;
    constexpr uint32_t kBlockSize = 2U;
    const std::array<float, kOrder * kOrder> custom = {0.5f, -0.25f, 0.75f, 1.f, 0.f, -0.5f, -0.75f, 0.25f, 0.5f};
    std::array<float, kOrder * kBlockSize> input = {1.f, -0.5f, 0.25f, 0.75f, -1.f, 0.5f};
    sfFDN::ScalarFeedbackMatrix matrix(
        {.source = sfFDN::MatrixData{kOrder, std::vector<float>(custom.begin(), custom.end())}});

    std::vector<float> actual_matrix(kOrder * kOrder);
    REQUIRE(matrix.GetMatrix(actual_matrix));
    REQUIRE(actual_matrix == std::vector<float>(custom.begin(), custom.end()));

    std::array<float, kOrder * kBlockSize> output{};
    sfFDN::AudioBuffer input_buffer(kBlockSize, kOrder, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kOrder, output);
    matrix.Process(input_buffer, output_buffer);
    RequireNear(output, DenseReference(custom, kOrder, kBlockSize, input));
}

TEST_CASE("DelayMatrix follows the row-major dest*N+src convention", "[feedback_matrix]")
{
    // Non-symmetric 3x3 case.
    // Matrix A (row-major, non-symmetric):
    //   A = [[1.0, 0.2, 0.3],
    //        [0.4, 1.0, 0.6],
    //        [0.7, 0.8, 1.0]]
    constexpr uint32_t N = 3;
    const std::vector<float> kGains = {1.0f, 0.2f, 0.3f, 0.4f, 1.0f, 0.6f, 0.7f, 0.8f, 1.0f};

    // delays[dest*N+src] (row=dest, col=src):
    //   dest=0: src=0→3, src=1→5, src=2→2
    //   dest=1: src=0→4, src=1→1, src=2→6
    //   dest=2: src=0→2, src=1→7, src=2→3
    constexpr std::array<uint32_t, N * N> kDelays = {3, 5, 2, 4, 1, 6, 2, 7, 3};

    sfFDN::ScalarFeedbackMatrix mixing_matrix({.source = sfFDN::MatrixData{N, kGains}});
    sfFDN::DelayMatrix delay_matrix(N, kDelays, mixing_matrix);

    // Stagger one impulse per source so every route contributes at a known sample.
    constexpr uint32_t kBlockSize = 12;
    std::array<float, N * kBlockSize> input{};
    std::array<float, N * kBlockSize> output{};

    for (auto source = 0u; source < N; ++source)
    {
        input[(source * kBlockSize) + source] = 1.f;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    std::size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter counter;
        delay_matrix.Process(input_buffer, output_buffer);
        allocations = counter.Count();
    }
    REQUIRE(allocations == 0);

    for (auto destination = 0u; destination < N; ++destination)
    {
        for (auto sample = 0u; sample < kBlockSize; ++sample)
        {
            float expected = 0.0f;
            for (auto source = 0u; source < N; ++source)
            {
                const auto arrival = source + kDelays[(destination * N) + source];
                if (sample == arrival)
                {
                    expected += kGains[(destination * N) + source];
                }
            }
            REQUIRE_THAT(output_buffer.GetChannelSpan(destination)[sample],
                         Catch::Matchers::WithinAbs(expected, 1e-6f));
        }
    }
}

TEST_CASE("FilterFeedbackMatrix repeats output after Clear", "[feedback_matrix]")
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

    float energy = 0.f;
    for (const float sample : output)
    {
        REQUIRE(std::isfinite(sample));
        energy += sample * sample;
    }
    REQUIRE(energy > 0.f);

    ffm->Clear();
    std::array<float, kMatSize * kBlockSize> repeated{};
    sfFDN::AudioBuffer repeated_buffer(kBlockSize, kMatSize, repeated);
    ffm->Process(input_buffer, repeated_buffer);
    for (const auto [actual, expected] : std::views::zip(repeated, output))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 1e-6f));
    }
}

TEST_CASE("Structured feedback matrices match dense processing without allocations", "[feedback_matrix]")
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
                sfFDN::ScalarFeedbackMatrix structured(
                    {.source = sfFDN::GeneratedMatrixOptions{.matrix_size = order, .generator = type}});
                sfFDN::ScalarFeedbackMatrix dense({.source = sfFDN::MatrixData{order, matrix_data}});

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

TEST_CASE("ScalarFeedbackMatrix SetMatrix disables structured processing", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 8;
    constexpr uint32_t kBlockSize = 3;
    const auto matrix_data = sfFDN::GenerateMatrix(kOrder, sfFDN::ScalarMatrixType::Random);
    sfFDN::ScalarFeedbackMatrix updated({.source = sfFDN::GeneratedMatrixOptions{
                                             .matrix_size = kOrder, .generator = sfFDN::ScalarMatrixType::Hadamard}});
    sfFDN::ScalarFeedbackMatrix dense({.source = sfFDN::MatrixData{kOrder, matrix_data}});
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

TEST_CASE("FilterFeedbackMatrix uses structured stage-zero processing", "[feedback_matrix]")
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
            .generator = type,
            .gain_per_samples = 1.f,
        });
        const auto matrix_data = sfFDN::GenerateMatrix(kOrder, type);
        sfFDN::ScalarFeedbackMatrix dense({.source = sfFDN::MatrixData{kOrder, matrix_data}});

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
// ---------------------------------------------------------------------------
// Tests added for matrix-order canonicalization (row-major: flat[row*N+col])
// ---------------------------------------------------------------------------

TEST_CASE("ScalarFeedbackMatrix uses row-major SetMatrix GetMatrix GetCoefficient and Process conventions",
          "[feedback_matrix]")
{
    // A = [[1,2,3],[4,5,6],[7,8,9]] stored in row-major flat order.
    constexpr uint32_t N = 3;
    const std::vector<float> kMatrix = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

    sfFDN::ScalarFeedbackMatrix mat({.source = sfFDN::MatrixData{N, kMatrix}});

    // GetCoefficient uses row-major A[row,col] = flat[row*N+col].
    REQUIRE(mat.GetCoefficient(0, 0) == 1.f);
    REQUIRE(mat.GetCoefficient(0, 1) == 2.f);
    REQUIRE(mat.GetCoefficient(0, 2) == 3.f);
    REQUIRE(mat.GetCoefficient(1, 0) == 4.f);
    REQUIRE(mat.GetCoefficient(1, 1) == 5.f);
    REQUIRE(mat.GetCoefficient(1, 2) == 6.f);
    REQUIRE(mat.GetCoefficient(2, 0) == 7.f);
    REQUIRE(mat.GetCoefficient(2, 1) == 8.f);
    REQUIRE(mat.GetCoefficient(2, 2) == 9.f);

    // GetMatrix returns the same row-major flat vector.
    std::vector<float> retrieved(N * N);
    REQUIRE(mat.GetMatrix(retrieved));
    REQUIRE(retrieved == kMatrix);

    // Process: y = A*x. Each standard basis vector x=e_k must produce column k of A.
    // x = e_0 = [1,0,0] → y = [A[0,0], A[1,0], A[2,0]] = [1, 4, 7]
    {
        std::array<float, N> in = {1.f, 0.f, 0.f};
        std::array<float, N> out{};
        sfFDN::AudioBuffer ib(1, N, in);
        sfFDN::AudioBuffer ob(1, N, out);
        mat.Process(ib, ob);
        REQUIRE_THAT(ob.GetChannelSpan(0)[0], Catch::Matchers::WithinAbs(1.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(1)[0], Catch::Matchers::WithinAbs(4.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(2)[0], Catch::Matchers::WithinAbs(7.f, 1e-5f));
    }
    // x = e_1 = [0,1,0] → y = [2, 5, 8]
    {
        std::array<float, N> in = {0.f, 1.f, 0.f};
        std::array<float, N> out{};
        sfFDN::AudioBuffer ib(1, N, in);
        sfFDN::AudioBuffer ob(1, N, out);
        mat.Process(ib, ob);
        REQUIRE_THAT(ob.GetChannelSpan(0)[0], Catch::Matchers::WithinAbs(2.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(1)[0], Catch::Matchers::WithinAbs(5.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(2)[0], Catch::Matchers::WithinAbs(8.f, 1e-5f));
    }
    // x = e_2 = [0,0,1] → y = [3, 6, 9]
    {
        std::array<float, N> in = {0.f, 0.f, 1.f};
        std::array<float, N> out{};
        sfFDN::AudioBuffer ib(1, N, in);
        sfFDN::AudioBuffer ob(1, N, out);
        mat.Process(ib, ob);
        REQUIRE_THAT(ob.GetChannelSpan(0)[0], Catch::Matchers::WithinAbs(3.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(1)[0], Catch::Matchers::WithinAbs(6.f, 1e-5f));
        REQUIRE_THAT(ob.GetChannelSpan(2)[0], Catch::Matchers::WithinAbs(9.f, 1e-5f));
    }

    // SetMatrix: replace with a different matrix and verify coefficients update.
    const std::vector<float> kMatrix2 = {9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
    REQUIRE(mat.SetMatrix(kMatrix2));
    REQUIRE(mat.GetCoefficient(0, 0) == 9.f);
    REQUIRE(mat.GetCoefficient(2, 2) == 1.f);
    REQUIRE(mat.GetCoefficient(0, 2) == 7.f);
    REQUIRE(mat.GetCoefficient(2, 0) == 3.f);
}

TEST_CASE("ScalarFeedbackMatrix SetMatrix rejects wrong size and leaves state unchanged", "[feedback_matrix]")
{
    constexpr uint32_t N = 3;
    const std::vector<float> kInit = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

    REQUIRE_THROWS_AS(sfFDN::MatrixData(N, std::vector<float>(N * N - 1, 0.f)), std::invalid_argument);

    sfFDN::ScalarFeedbackMatrix mat({.source = sfFDN::MatrixData{N, kInit}});

    // One element short.
    std::vector<float> short_vec(N * N - 1, 0.f);
    REQUIRE_FALSE(mat.SetMatrix(short_vec));
    // State must be unchanged.
    REQUIRE(mat.GetCoefficient(0, 0) == 1.f);
    REQUIRE(mat.GetCoefficient(2, 2) == 9.f);

    // One element long.
    std::vector<float> long_vec(N * N + 1, 0.f);
    REQUIRE_FALSE(mat.SetMatrix(long_vec));
    REQUIRE(mat.GetCoefficient(0, 0) == 1.f);
    REQUIRE(mat.GetCoefficient(2, 2) == 9.f);

    // Zero size.
    REQUIRE_FALSE(mat.SetMatrix(std::span<const float>{}));
    REQUIRE(mat.GetCoefficient(0, 0) == 1.f);

    // A different NxN size (e.g. 4x4=16 elements for a 3x3 matrix).
    std::vector<float> wrong_order(16, 0.f);
    REQUIRE_FALSE(mat.SetMatrix(wrong_order));
    REQUIRE(mat.GetCoefficient(0, 0) == 1.f);

    // Correct size is accepted.
    std::vector<float> correct(N * N, 99.f);
    REQUIRE(mat.SetMatrix(correct));
    REQUIRE(mat.GetCoefficient(1, 1) == 99.f);
}

TEST_CASE("FilterFeedbackMatrix GetFirstMatrix returns row-major layout", "[feedback_matrix]")
{
    // Use a non-structured matrix type (Random) with stage_count=0 so that
    // Process immediately applies matrix_[0] with no delays (stateless path).
    constexpr uint32_t N = 4;
    sfFDN::CascadedFeedbackMatrixOptions opts{
        .matrix_size = N,
        .stage_count = 0,
        .sparsity = 1.f,
        .generator = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 1.f,
    };
    sfFDN::FilterFeedbackMatrix ffm(opts);

    std::vector<float> first_mat(N * N);
    REQUIRE(ffm.GetFirstMatrix(first_mat));

    // Verify layout: for each standard basis input e_k (only channel k is 1, rest 0),
    // Process output[dest][0] must equal first_mat[dest*N+k] (row-major A[dest,k]).
    // ffm has no delay banks (stage_count=0), so it is stateless across calls.
    for (auto k = 0u; k < N; ++k)
    {
        std::array<float, N> in_data{};
        in_data[k] = 1.f;
        std::array<float, N> out_data{};
        sfFDN::AudioBuffer ib(1, N, in_data);
        sfFDN::AudioBuffer ob(1, N, out_data);
        ffm.Process(ib, ob);
        for (auto dest = 0u; dest < N; ++dest)
        {
            REQUIRE_THAT(ob.GetChannelSpan(dest)[0], Catch::Matchers::WithinAbs(first_mat[dest * N + k], 1e-5f));
        }
    }

    // GetFirstMatrix must fail on wrong-size span.
    std::vector<float> wrong(N * N - 1);
    REQUIRE_FALSE(ffm.GetFirstMatrix(wrong));
}

TEST_CASE("FilterFeedbackMatrix reproduces seeded random-family cascades", "[feedback_matrix]")
{
    // Covers both a random-matrix generator (where the seed changes the matrix itself) and a random-recipe
    // generator (RandomHouseholder, where the seed still drives the cascade even at the uint32_t boundary),
    // parameterized by generator/seed rather than duplicating the same first/repeated/different pattern per case.
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kBlockSize = 16U;
    constexpr uint32_t kBlockCount = 24U;

    struct Case
    {
        sfFDN::ScalarMatrixType generator;
        float sparsity;
        float gain_per_samples;
        uint32_t seed;
        uint32_t different_seed;
    };

    constexpr std::array<Case, 2> kCases = {{
        {sfFDN::ScalarMatrixType::Random, 2.5f, 0.98f, 0x5EED1234U, 0x5EED1235U},
        {sfFDN::ScalarMatrixType::RandomHouseholder, 2.f, 0.99f, std::numeric_limits<uint32_t>::max(), 0U},
    }};

    for (const auto& test_case : kCases)
    {
        const sfFDN::CascadedFeedbackMatrixOptions options = {
            .matrix_size = kOrder,
            .stage_count = 2U,
            .sparsity = test_case.sparsity,
            .generator = test_case.generator,
            .gain_per_samples = test_case.gain_per_samples,
            .rng_seed = test_case.seed,
        };
        sfFDN::FilterFeedbackMatrix first(options);
        sfFDN::FilterFeedbackMatrix repeated(options);
        auto different_options = options;
        different_options.rng_seed = test_case.different_seed;
        sfFDN::FilterFeedbackMatrix different(different_options);

        std::vector<float> first_matrix(kOrder * kOrder);
        std::vector<float> repeated_matrix(kOrder * kOrder);
        std::vector<float> different_matrix(kOrder * kOrder);
        REQUIRE(first.GetFirstMatrix(first_matrix));
        REQUIRE(repeated.GetFirstMatrix(repeated_matrix));
        REQUIRE(different.GetFirstMatrix(different_matrix));
        REQUIRE(first_matrix == repeated_matrix);
        REQUIRE(first_matrix != different_matrix);

        const auto first_output = RenderCascade(first, kBlockSize, kBlockCount);
        const auto repeated_output = RenderCascade(repeated, kBlockSize, kBlockCount);
        const auto different_output = RenderCascade(different, kBlockSize, kBlockCount);
        INFO("generator=" << static_cast<int>(test_case.generator) << " seed=" << test_case.seed);
        REQUIRE(std::ranges::any_of(first_output, [](float sample) { return sample != 0.f; }));
        RequireNear(repeated_output, first_output);
        REQUIRE(different_output != first_output);
    }
}

TEST_CASE("FilterFeedbackMatrix repeats default and zero-seed cascade recipes", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kBlockSize = 16U;
    constexpr uint32_t kBlockCount = 24U;
    constexpr std::array kGenerators = {
        sfFDN::ScalarMatrixType::Random,
        sfFDN::ScalarMatrixType::Hadamard,
        sfFDN::ScalarMatrixType::Householder,
    };

    for (const auto generator : kGenerators)
    {
        const sfFDN::CascadedFeedbackMatrixOptions default_options = {
            .matrix_size = kOrder,
            .stage_count = 2U,
            .sparsity = 2.5f,
            .generator = generator,
            .gain_per_samples = 0.98f,
        };
        auto zero_options = default_options;
        zero_options.rng_seed = 0U;
        sfFDN::FilterFeedbackMatrix default_first(default_options);
        sfFDN::FilterFeedbackMatrix default_repeated(default_options);
        sfFDN::FilterFeedbackMatrix zero_first(zero_options);
        sfFDN::FilterFeedbackMatrix zero_repeated(zero_options);
        std::vector<float> default_matrix(kOrder * kOrder);
        std::vector<float> default_repeated_matrix(kOrder * kOrder);
        std::vector<float> zero_matrix(kOrder * kOrder);
        std::vector<float> zero_repeated_matrix(kOrder * kOrder);
        REQUIRE(default_first.GetFirstMatrix(default_matrix));
        REQUIRE(default_repeated.GetFirstMatrix(default_repeated_matrix));
        REQUIRE(zero_first.GetFirstMatrix(zero_matrix));
        REQUIRE(zero_repeated.GetFirstMatrix(zero_repeated_matrix));
        REQUIRE(default_matrix == default_repeated_matrix);
        REQUIRE(zero_matrix == zero_repeated_matrix);

        const auto default_output = RenderCascade(default_first, kBlockSize, kBlockCount);
        const auto default_repeated_output = RenderCascade(default_repeated, kBlockSize, kBlockCount);
        const auto zero_output = RenderCascade(zero_first, kBlockSize, kBlockCount);
        const auto zero_repeated_output = RenderCascade(zero_repeated, kBlockSize, kBlockCount);
        INFO("generator=" << static_cast<int>(generator));
        RequireNear(default_repeated_output, default_output);
        RequireNear(zero_repeated_output, zero_output);

        if (generator == sfFDN::ScalarMatrixType::Random)
        {
            REQUIRE(default_matrix != zero_matrix);
            REQUIRE(default_output != zero_output);
        }
    }
}

TEST_CASE("FilterFeedbackMatrix reproduces seeded structured cascade delays", "[feedback_matrix]")
{
    constexpr uint32_t kOrder = 4U;
    constexpr uint32_t kBlockSize = 16U;
    constexpr uint32_t kBlockCount = 24U;
    constexpr std::array kTypes = {sfFDN::ScalarMatrixType::Hadamard, sfFDN::ScalarMatrixType::Householder};

    for (const auto type : kTypes)
    {
        const sfFDN::CascadedFeedbackMatrixOptions options = {
            .matrix_size = kOrder,
            .stage_count = 2U,
            .sparsity = 3.f,
            .generator = type,
            .gain_per_samples = 1.f,
            .rng_seed = 0xA11CE55U,
        };
        sfFDN::FilterFeedbackMatrix first(options);
        sfFDN::FilterFeedbackMatrix repeated(options);
        auto different_options = options;
        different_options.rng_seed = 0xA11CE56U;
        sfFDN::FilterFeedbackMatrix different(different_options);
        std::vector<float> first_matrix(kOrder * kOrder);
        std::vector<float> different_matrix(kOrder * kOrder);
        REQUIRE(first.GetFirstMatrix(first_matrix));
        REQUIRE(different.GetFirstMatrix(different_matrix));
        REQUIRE(first_matrix == different_matrix);

        const auto first_output = RenderCascade(first, kBlockSize, kBlockCount);
        const auto repeated_output = RenderCascade(repeated, kBlockSize, kBlockCount);
        const auto different_output = RenderCascade(different, kBlockSize, kBlockCount);
        INFO("type=" << static_cast<int>(type));
        REQUIRE(std::ranges::any_of(first_output, [](float sample) { return sample != 0.f; }));
        REQUIRE(std::ranges::any_of(different_output, [](float sample) { return sample != 0.f; }));
        RequireNear(repeated_output, first_output);
        REQUIRE(different_output != first_output);
    }
}
