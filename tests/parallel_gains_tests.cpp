#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include <sndfile.h>

#include "sffdn/parallel_gains.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "test_utils.h"

TEST_CASE("ParallelGains Split applies configured gains to each output channel", "[parallel_gains]")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.25f, 0.5f, 0.75f, 1.f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Split);
    parallel_gains.SetGains(kGains);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kChannelCount * kBlockSize, 0.f);
    for (auto i = 0u; i < kBlockSize; ++i)
    {
        input[i] = i;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 0.25, 0.5, 0.75, 1,   1.25, 1.5, 1.75, 2,   2.25, 0,   0.5,  1,   1.5,
                                       2, 2.5,  3,   3.5,  4,   4.5,  0,   0.75, 1.5, 2.25, 3,   3.75, 4.5, 5.25,
                                       6, 6.75, 0,   1.f,  2.f, 3.f,  4.f, 5.f,  6.f, 7.f,  8.f, 9.f};

    REQUIRE(output.size() == expected_out.size());
    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("ParallelGains Merge sums scaled input channels", "[parallel_gains]")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, kChannelCount> kGains = {0.5f, 0.5f, 0.5f, 0.5f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Merge);
    parallel_gains.SetGains(kGains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);

    for (auto i = 0u; i < kChannelCount; ++i)
    {
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            input[i * kBlockSize + j] = j;
        }
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    REQUIRE(output.size() == expected_out.size());

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

// With frequency and amplitude to 0, this should behave the same as a normal ParallelGain
TEST_CASE("TimeVaryingParallelGains Split matches ParallelGains without modulation", "[parallel_gains]")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    const std::vector<float> kGains = {0.25f, 0.5f, 0.75f, 1.f};
    sfFDN::ParallelGainsOptions gains_options;
    gains_options.mode = sfFDN::ParallelGainsMode::Split;
    gains_options.gains = kGains;
    sfFDN::TimeVaryingParallelGains tv_parallel_gains(gains_options);
    tv_parallel_gains.SetCenterGains(kGains);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = i;
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    tv_parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out(kChannelCount * kBlockSize, 0.f);
    sfFDN::AudioBuffer expected_out_buffer(kBlockSize, kChannelCount, expected_out);
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Split);
    parallel_gains.SetGains(kGains);
    parallel_gains.Process(input_buffer, expected_out_buffer);

    REQUIRE(output.size() == expected_out.size());
    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("TimeVaryingParallelGains Merge matches ParallelGains without modulation", "[parallel_gains]")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 10;
    const std::vector<float> kGains = {0.5f, 0.5f, 0.5f, 0.5f};
    sfFDN::ParallelGainsOptions gains_options;
    gains_options.mode = sfFDN::ParallelGainsMode::Merge;
    gains_options.gains = kGains;
    sfFDN::TimeVaryingParallelGains tv_parallel_gains(gains_options);
    tv_parallel_gains.SetCenterGains(kGains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    for (auto chan_idx = 0u; chan_idx < input_buffer.ChannelCount(); ++chan_idx)
    {
        auto channel = input_buffer.GetChannelSpan(chan_idx);
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            channel[j] = j;
        }
    }

    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
    tv_parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out(kBlockSize, 0.f);
    sfFDN::AudioBuffer expected_out_buffer(kBlockSize, 1, expected_out);
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Merge);
    parallel_gains.SetGains(kGains);
    parallel_gains.Process(input_buffer, expected_out_buffer);

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(expected_out[i]));
    }
}

TEST_CASE("TimeVaryingParallelGains Split modulates gains and Clear restores initial phase", "[parallel_gains]")
{
    constexpr std::array<float, 4> kExpectedGain = {0.5f, 0.6767767f, 0.75f, 0.6767767f};
    constexpr std::array<float, 4> kContinuedGain = {0.5f, 0.3232233f, 0.25f, 0.3232233f};
    const sfFDN::ParallelGainsOptions options{
        .mode = sfFDN::ParallelGainsMode::Split,
        .gains = {0.5f},
        .time_varying_config = {{.frequency = 0.125f, .amplitude = 0.25f, .initial_phase = 0.f}},
    };
    sfFDN::TimeVaryingParallelGains gains(options);
    std::array<float, 4> input = {1.f, 1.f, 1.f, 1.f};
    std::array<float, 4> output{};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    gains.Process(input_buffer, output_buffer);

    REQUIRE(gains.InputChannelCount() == 1);
    REQUIRE(gains.OutputChannelCount() == 1);
    for (size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(kExpectedGain[i]));
    }

    auto continued = gains.Clone();
    std::array<float, 4> continued_output{};
    sfFDN::AudioBuffer continued_output_buffer(continued_output);
    continued->Process(input_buffer, continued_output_buffer);
    for (size_t i = 0; i < continued_output.size(); ++i)
    {
        REQUIRE(continued_output[i] == Catch::Approx(kContinuedGain[i]));
    }

    gains.Clear();
    std::array<float, 4> reset_output{};
    sfFDN::AudioBuffer reset_output_buffer(reset_output);
    gains.Process(input_buffer, reset_output_buffer);
    for (size_t i = 0; i < reset_output.size(); ++i)
    {
        REQUIRE(reset_output[i] == Catch::Approx(kExpectedGain[i]));
    }
}

TEST_CASE("TimeVaryingParallelGains processes Merge and Parallel modes without allocation", "[parallel_gains]")
{
    const sfFDN::ParallelGainsOptions merge_options{
        .mode = sfFDN::ParallelGainsMode::Merge,
        .gains = {0.5f, 1.f},
        .time_varying_config = {{.frequency = 0.125f, .amplitude = 0.25f, .initial_phase = 0.f},
                                {.frequency = 0.125f, .amplitude = 0.5f, .initial_phase = 0.25f}},
    };
    sfFDN::TimeVaryingParallelGains merge(merge_options);
    std::array<float, 8> merge_input = {1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f};
    std::array<float, 4> merge_output{};
    sfFDN::AudioBuffer const merge_input_buffer(4, 2, merge_input);
    sfFDN::AudioBuffer merge_output_buffer(merge_output);
    merge.Process(merge_input_buffer, merge_output_buffer);
    constexpr std::array<float, 4> kExpectedMerge = {3.5f, 3.3838835f, 2.75f, 1.9696699f};
    for (size_t i = 0; i < merge_output.size(); ++i)
    {
        REQUIRE(merge_output[i] == Catch::Approx(kExpectedMerge[i]));
    }

    const sfFDN::ParallelGainsOptions parallel_options{
        .mode = sfFDN::ParallelGainsMode::Parallel,
        .gains = {0.5f, 1.f},
        .time_varying_config = merge_options.time_varying_config,
    };
    sfFDN::TimeVaryingParallelGains parallel(parallel_options);
    std::array<float, 8> parallel_output{};
    sfFDN::AudioBuffer parallel_output_buffer(4, 2, parallel_output);
    parallel.Process(merge_input_buffer, parallel_output_buffer);
    constexpr std::array<float, 4> kExpectedFirst = {0.5f, 0.6767767f, 0.75f, 0.6767767f};
    constexpr std::array<float, 4> kExpectedSecond = {3.f, 2.7071068f, 2.f, 1.2928932f};
    for (size_t i = 0; i < kExpectedFirst.size(); ++i)
    {
        REQUIRE(parallel_output[i] == Catch::Approx(kExpectedFirst[i]));
        REQUIRE(parallel_output[i + 4] == Catch::Approx(kExpectedSecond[i]));
    }

    auto clone = parallel.Clone();
    std::array<float, 8> clone_output{};
    sfFDN::AudioBuffer clone_output_buffer(4, 2, clone_output);
    clone->Process(merge_input_buffer, clone_output_buffer);
    std::array<float, 8> continued_output{};
    sfFDN::AudioBuffer continued_output_buffer(4, 2, continued_output);
    parallel.Process(merge_input_buffer, continued_output_buffer);
    for (size_t i = 0; i < clone_output.size(); ++i)
    {
        REQUIRE(clone_output[i] == Catch::Approx(continued_output[i]));
    }

    {
        sfFDNTest::ScopedAllocationCounter const allocation_counter;
        parallel.Process(merge_input_buffer, parallel_output_buffer);
        REQUIRE(allocation_counter.Count() == 0);
    }
}

TEST_CASE("MakeParallelGainsFromConfig selects static and time-varying implementations", "[parallel_gains]")
{
    SECTION("static modes continue to construct ParallelGains")
    {
        for (const auto mode :
             {sfFDN::ParallelGainsMode::Split, sfFDN::ParallelGainsMode::Merge, sfFDN::ParallelGainsMode::Parallel})
        {
            const sfFDN::ParallelGainsOptions static_options{
                .mode = mode, .gains = {2.f, 3.f}, .time_varying_config = {}};
            const auto static_gains = sfFDN::MakeParallelGainsFromConfig(static_options);
            REQUIRE(dynamic_cast<sfFDN::ParallelGains*>(static_gains.get()) != nullptr);
            REQUIRE(static_gains->InputChannelCount() == (mode == sfFDN::ParallelGainsMode::Split ? 1U : 2U));
            REQUIRE(static_gains->OutputChannelCount() == (mode == sfFDN::ParallelGainsMode::Merge ? 1U : 2U));
        }
    }

    SECTION("Merge still accumulates and Parallel still supports in-place scaling")
    {
        const sfFDN::ParallelGainsOptions merge_options{
            .mode = sfFDN::ParallelGainsMode::Merge, .gains = {2.f, 3.f}, .time_varying_config = {}};
        const auto merge = sfFDN::MakeParallelGainsFromConfig(merge_options);
        std::array<float, 2> merge_input = {4.F, 5.F};
        std::array<float, 1> merge_output = {7.F};
        const sfFDN::AudioBuffer merge_input_buffer(1U, 2U, merge_input);
        sfFDN::AudioBuffer merge_output_buffer(merge_output);
        merge->Process(merge_input_buffer, merge_output_buffer);
        REQUIRE(merge_output[0] == 30.F);

        const sfFDN::ParallelGainsOptions parallel_options{
            .mode = sfFDN::ParallelGainsMode::Parallel, .gains = {2.f, 3.f}, .time_varying_config = {}};
        const auto parallel = sfFDN::MakeParallelGainsFromConfig(parallel_options);
        std::array<float, 2> in_place = {4.F, 5.F};
        sfFDN::AudioBuffer in_place_buffer(1U, 2U, in_place);
        parallel->Process(in_place_buffer, in_place_buffer);
        REQUIRE(in_place == std::array{8.F, 15.F});
    }

    const sfFDN::ParallelGainsOptions time_varying_options{
        .mode = sfFDN::ParallelGainsMode::Split,
        .gains = {1.f, 1.f},
        .time_varying_config =
            {
                {.frequency = 0.f, .amplitude = 0.f, .initial_phase = 0.f},
                {.frequency = 0.f, .amplitude = 0.f, .initial_phase = 0.f},
            },
    };
    const auto time_varying_gains = sfFDN::MakeParallelGainsFromConfig(time_varying_options);
    REQUIRE(dynamic_cast<sfFDN::TimeVaryingParallelGains*>(time_varying_gains.get()) != nullptr);
    REQUIRE(time_varying_gains->InputChannelCount() == 1);
    REQUIRE(time_varying_gains->OutputChannelCount() == 2);

    const sfFDN::ParallelGainsOptions empty_gains{
        .mode = sfFDN::ParallelGainsMode::Split, .gains = {}, .time_varying_config = {}};
    const auto invalid_gains = sfFDN::MakeParallelGainsFromConfig(empty_gains);
    REQUIRE(invalid_gains->InputChannelCount() == 1);
    REQUIRE(invalid_gains->OutputChannelCount() == 0);
}