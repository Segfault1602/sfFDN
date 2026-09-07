#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <stdexcept>

#include "allocation_counter.h"
#include "sffdn/channel_matrix.h"
#include "sffdn/parallel_gains.h"

TEST_CASE("ChannelMatrix applies row-major rectangular coefficients", "[channel_matrix]")
{
    sfFDN::ChannelMatrix matrix({
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = {1.F, 2.F, -1.F, 0.5F, 3.F, -2.F},
    });
    std::array<float, 4> input = {1.F, 2.F, 10.F, 20.F};
    std::array<float, 6> output = {99.F, 99.F, 99.F, 99.F, 99.F, 99.F};
    const sfFDN::AudioBuffer input_buffer(2U, 2U, input);
    sfFDN::AudioBuffer output_buffer(2U, 3U, output);

    matrix.Process(input_buffer, output_buffer);

    REQUIRE(output == std::array{21.F, 42.F, 4.F, 8.F, -17.F, -34.F});
}

TEST_CASE("ChannelMatrix matches ParallelGains for degenerate shapes", "[channel_matrix]")
{
    constexpr std::array<float, 3> kGains = {2.F, -3.F, 0.5F};
    std::array<float, 4> mono_input = {1.F, 2.F, 3.F, 4.F};
    std::array<float, 12> matrix_split_output{};
    std::array<float, 12> gains_split_output{};
    const sfFDN::AudioBuffer mono_input_buffer(mono_input);
    sfFDN::AudioBuffer matrix_split_buffer(4U, 3U, matrix_split_output);
    sfFDN::AudioBuffer gains_split_buffer(4U, 3U, gains_split_output);

    sfFDN::ChannelMatrix split_matrix({
        .input_channel_count = 1U,
        .output_channel_count = 3U,
        .coefficients = {2.F, -3.F, 0.5F},
    });
    sfFDN::ParallelGains split_gains(sfFDN::ParallelGainsMode::Split, kGains);
    split_matrix.Process(mono_input_buffer, matrix_split_buffer);
    split_gains.Process(mono_input_buffer, gains_split_buffer);
    REQUIRE(matrix_split_output == gains_split_output);

    std::array<float, 12> multichannel_input = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F, 10.F, 11.F, 12.F};
    std::array<float, 4> matrix_merge_output{};
    std::array<float, 4> gains_merge_output{};
    const sfFDN::AudioBuffer multichannel_input_buffer(4U, 3U, multichannel_input);
    sfFDN::AudioBuffer matrix_merge_buffer(matrix_merge_output);
    sfFDN::AudioBuffer gains_merge_buffer(gains_merge_output);

    sfFDN::ChannelMatrix merge_matrix({
        .input_channel_count = 3U,
        .output_channel_count = 1U,
        .coefficients = {2.F, -3.F, 0.5F},
    });
    sfFDN::ParallelGains merge_gains(sfFDN::ParallelGainsMode::Merge, kGains);
    merge_matrix.Process(multichannel_input_buffer, matrix_merge_buffer);
    merge_gains.Process(multichannel_input_buffer, gains_merge_buffer);
    REQUIRE(matrix_merge_output == gains_merge_output);
}

TEST_CASE("ChannelMatrix clones configuration and clears without changing it", "[channel_matrix]")
{
    sfFDN::ChannelMatrix matrix({
        .input_channel_count = 2U,
        .output_channel_count = 2U,
        .coefficients = {1.F, 2.F, 3.F, 4.F},
    });
    auto clone = matrix.Clone();
    clone->Clear();

    REQUIRE(clone->InputChannelCount() == 2U);
    REQUIRE(clone->OutputChannelCount() == 2U);

    std::array<float, 4> input = {1.F, 2.F, 3.F, 4.F};
    std::array<float, 4> original_output{};
    std::array<float, 4> clone_output{};
    const sfFDN::AudioBuffer input_buffer(2U, 2U, input);
    sfFDN::AudioBuffer original_output_buffer(2U, 2U, original_output);
    sfFDN::AudioBuffer clone_output_buffer(2U, 2U, clone_output);
    matrix.Process(input_buffer, original_output_buffer);
    clone->Process(input_buffer, clone_output_buffer);
    REQUIRE(clone_output == original_output);
}

TEST_CASE("ChannelMatrix rejects invalid dimensions and coefficient counts", "[channel_matrix]")
{
    REQUIRE_THROWS_AS(sfFDN::ChannelMatrix({.input_channel_count = 0U, .output_channel_count = 1U, .coefficients = {}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::ChannelMatrix({.input_channel_count = 1U, .output_channel_count = 0U, .coefficients = {}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::ChannelMatrix({.input_channel_count = 2U, .output_channel_count = 3U, .coefficients = {1.F}}),
        std::invalid_argument);
}

TEST_CASE("ChannelMatrix processing is allocation-free", "[channel_matrix]")
{
    sfFDN::ChannelMatrix matrix({
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F},
    });
    std::array<float, 16> input{};
    std::array<float, 24> output{};
    const sfFDN::AudioBuffer input_buffer(8U, 2U, input);
    sfFDN::AudioBuffer output_buffer(8U, 3U, output);
    matrix.Process(input_buffer, output_buffer);

    const sfFDNTest::ScopedAllocationCounter allocation_counter;
    matrix.Process(input_buffer, output_buffer);
    REQUIRE(allocation_counter.Count() == 0U);
}
