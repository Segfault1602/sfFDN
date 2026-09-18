// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "allocation_counter.h"
#include "sffdn/sffdn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

TEST_CASE("FilterBank constructs every single-channel processor option", "[filter]")
{
    const sfFDN::FilterDesigner designer(48000.F);
    const sfFDN::GraphicEQOptions graphic_eq{
        .gains_db = {6.F, -3.F, 4.F, -2.F, 1.F, 0.F, -1.F, 2.F, -4.F, 3.F},
        .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
    };
    const sfFDN::TimeVaryingSchroederAllpassSectionOptions time_varying_allpass{
        .delays = {3.F},
        .gains = {0.25F},
        .time_varying_config = {{.frequency = 0.01F, .amplitude = 0.1F, .initial_phase = 0.F}},
    };
    const sfFDN::MultichannelProcessorOptions options{
        .channels =
            {
                sfFDN::SchroederAllpassSectionOptions{.delays = {2.F}, .gains = {0.25F}},
                time_varying_allpass,
                sfFDN::AllpassFilterOptions{.coeff = 0.25F},
                sfFDN::CascadedBiquadsOptions{.coeffs = {{0.5F, 0.25F, 0.F, 1.F, -0.3F, 0.F}}},
                sfFDN::FirOptions{.coeffs = {0.25F, -0.5F, 0.75F, 0.125F}},
                sfFDN::FirOptions{.coeffs = {0.5F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, -0.25F}},
                sfFDN::DelayOptions{.delay = 5.F, .max_delay = 8U},
                sfFDN::DelayOptions{
                    .delay = 3.F,
                    .max_delay = 5U,
                    .interp_type = sfFDN::DelayInterpolationType::Linear,
                    .lfo_config =
                        sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = 0.25F, .initial_phase = 0.F}},
                graphic_eq,
                sfFDN::DattorroDelayOptions{.delay_config = {.delay = 7.F, .max_delay = 16U}},
                sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.5F, .dc_block = false},
                sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5F},
                sfFDN::RingModulatorOptions{.frequency = 0.01F, .amplitude = 1.F, .initial_phase = 0.F},
            },
    };

    sfFDN::FilterBank bank(options, designer);
    REQUIRE(bank.InputChannelCount() == options.channels.size());
    REQUIRE(bank.OutputChannelCount() == options.channels.size());

    auto reference = std::make_unique<sfFDN::FilterBank>();
    reference->AddFilter(std::make_unique<sfFDN::SchroederAllpassSection>(
        std::get<sfFDN::SchroederAllpassSectionOptions>(options.channels[0].value())));
    reference->AddFilter(std::make_unique<sfFDN::TimeVaryingSchroederAllpassSection>(
        std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(options.channels[1].value())));
    reference->AddFilter(
        std::make_unique<sfFDN::AllpassFilter>(std::get<sfFDN::AllpassFilterOptions>(options.channels[2].value())));
    auto biquads = std::make_unique<sfFDN::CascadedBiquads>();
    biquads->SetCoefficients(std::get<sfFDN::CascadedBiquadsOptions>(options.channels[3].value()).coeffs);
    reference->AddFilter(std::move(biquads));
    reference->AddFilter(
        std::make_unique<sfFDN::Fir>(std::get<sfFDN::FirOptions>(options.channels[4].value())));
    reference->AddFilter(std::make_unique<sfFDN::SparseFir>(sfFDN::SparseFirOptions{
        .coeffs = {{0U, 0.5F}, {8U, -0.25F}},
    }));
    reference->AddFilter(
        std::make_unique<sfFDN::DelayInterp>(std::get<sfFDN::DelayOptions>(options.channels[6].value())));
    reference->AddFilter(
        std::make_unique<sfFDN::DelayTimeVarying>(std::get<sfFDN::DelayOptions>(options.channels[7].value())));
    auto graphic_eq_filter = std::make_unique<sfFDN::CascadedBiquads>();
    graphic_eq_filter->SetCoefficients(
        designer.DesignFilter(std::get<sfFDN::GraphicEQOptions>(options.channels[8].value())));
    reference->AddFilter(std::move(graphic_eq_filter));
    reference->AddFilter(
        std::make_unique<sfFDN::DattorroDelay>(std::get<sfFDN::DattorroDelayOptions>(options.channels[9].value())));
    reference->AddFilter(std::make_unique<sfFDN::ControllableFullWaveRectifier>(
        std::get<sfFDN::ControllableFullWaveRectifierOptions>(options.channels[10].value())));
    reference->AddFilter(std::make_unique<sfFDN::SignalDependentFractionalDelay>(
        std::get<sfFDN::SignalDependentFractionalDelayOptions>(options.channels[11].value())));
    reference->AddFilter(
        std::make_unique<sfFDN::RingModulator>(std::get<sfFDN::RingModulatorOptions>(options.channels[12].value())));

    constexpr uint32_t kSamples = 128;
    std::vector<float> input(options.channels.size() * kSamples);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(static_cast<int>(i % 7U) - 3) * 0.125F;
    }
    std::vector<float> actual(input.size(), 0.F);
    std::vector<float> expected(input.size(), 0.F);
    sfFDN::AudioBuffer input_buffer(kSamples, options.channels.size(), input);
    sfFDN::AudioBuffer actual_buffer(kSamples, options.channels.size(), actual);
    sfFDN::AudioBuffer expected_buffer(kSamples, options.channels.size(), expected);
    bank.Process(input_buffer, actual_buffer);
    reference->Process(input_buffer, expected_buffer);
    REQUIRE(actual == expected);

}

TEST_CASE("FilterBank handles mixed processors, bypasses, cloning, and in-place processing", "[filter]")
{
    constexpr uint32_t kSamples = 8;
    const sfFDN::FilterDesigner designer(sfFDN::kDefaultSampleRate);
    const sfFDN::MultichannelProcessorOptions options{
        .channels =
            {
                sfFDN::FirOptions{.coeffs = {0.5F, 0.25F, -0.125F}},
                std::nullopt,
                sfFDN::DelayOptions{.delay = 12.F, .max_delay = 24U},
                sfFDN::AllpassFilterOptions{.coeff = 0.25F},
            },
    };
    sfFDN::FilterBank in_place_bank(options, designer);
    sfFDN::FilterBank out_of_place_bank(options, designer);
    std::vector<float> samples(4U * kSamples, 0.F);
    samples[0] = 1.F;
    samples[kSamples] = 0.75F;
    samples[2U * kSamples] = 0.5F;
    samples[3U * kSamples] = 0.25F;
    auto original = samples;

    sfFDN::AudioBuffer in_place_buffer(kSamples, 4U, samples);
    std::vector<float> out_of_place(samples.size(), 0.F);
    sfFDN::AudioBuffer original_buffer(kSamples, 4U, original);
    sfFDN::AudioBuffer out_of_place_buffer(kSamples, 4U, out_of_place);
    in_place_bank.Process(in_place_buffer, in_place_buffer);
    out_of_place_bank.Process(original_buffer, out_of_place_buffer);
    for (size_t sample = 0; sample < samples.size(); ++sample)
    {
        REQUIRE_THAT(samples[sample], Catch::Matchers::WithinAbs(out_of_place[sample], 1.e-6F));
    }

    sfFDN::FilterBank partitioned_bank(options, designer);
    std::vector<float> partitioned_output(samples.size(), 0.F);
    sfFDN::AudioBuffer partitioned_buffer(kSamples, 4U, partitioned_output);
    for (uint32_t offset = 0; offset < kSamples; offset += 4U)
    {
        const auto input_chunk = original_buffer.Offset(offset, 4U);
        auto output_chunk = partitioned_buffer.Offset(offset, 4U);
        partitioned_bank.Process(input_chunk, output_chunk);
    }
    REQUIRE(partitioned_output == out_of_place);

    sfFDN::FilterBank bank(options, designer);
    std::vector<float> priming_input(4U * kSamples, 0.F);
    priming_input[2U * kSamples] = 1.F;
    std::vector<float> priming_output(priming_input.size(), 0.F);
    sfFDN::AudioBuffer priming_input_buffer(kSamples, 4U, priming_input);
    sfFDN::AudioBuffer priming_output_buffer(kSamples, 4U, priming_output);
    bank.Process(priming_input_buffer, priming_output_buffer);
    auto clone = bank.Clone();
    std::vector<float> input(4U * kSamples, 0.F);
    std::vector<float> bank_output(input.size(), 0.F);
    std::vector<float> clone_output(input.size(), 0.F);
    sfFDN::AudioBuffer input_buffer(kSamples, 4U, input);
    sfFDN::AudioBuffer bank_output_buffer(kSamples, 4U, bank_output);
    sfFDN::AudioBuffer clone_output_buffer(kSamples, 4U, clone_output);
    bank.Process(input_buffer, bank_output_buffer);
    clone->Process(input_buffer, clone_output_buffer);
    for (size_t sample = 0; sample < bank_output.size(); ++sample)
    {
        REQUIRE_THAT(bank_output[sample], Catch::Matchers::WithinAbs(clone_output[sample], 1.e-6F));
    }

    sfFDN::FilterBank fresh(options, designer);
    std::vector<float> fresh_output(input.size(), 0.F);
    sfFDN::AudioBuffer fresh_output_buffer(kSamples, 4U, fresh_output);
    fresh.Process(input_buffer, fresh_output_buffer);
    REQUIRE(std::ranges::any_of(bank_output, [](float sample) { return std::abs(sample) > 1.e-6F; }));
    REQUIRE(bank_output != fresh_output);

    std::fill(priming_output.begin(), priming_output.end(), 0.F);
    bank.Process(priming_input_buffer, priming_output_buffer);
    bank.Clear();
    std::fill(bank_output.begin(), bank_output.end(), 0.F);
    std::fill(fresh_output.begin(), fresh_output.end(), 0.F);
    bank.Process(input_buffer, bank_output_buffer);
    fresh.Clear();
    fresh.Process(input_buffer, fresh_output_buffer);
    for (size_t sample = 0; sample < bank_output.size(); ++sample)
    {
        REQUIRE_THAT(bank_output[sample], Catch::Matchers::WithinAbs(fresh_output[sample], 1.e-6F));
    }

    bank.Process(input_buffer, bank_output_buffer);
    const sfFDNTest::ScopedAllocationCounter counter;
    bank.Process(input_buffer, bank_output_buffer);
    REQUIRE(counter.Count() == 0);
}

TEST_CASE("FilterBank accepts empty and all-bypass options", "[filter]")
{
    const sfFDN::FilterDesigner designer(sfFDN::kDefaultSampleRate);
    sfFDN::FilterBank empty(sfFDN::MultichannelProcessorOptions{}, designer);
    REQUIRE(empty.InputChannelCount() == 0U);

    const sfFDN::MultichannelProcessorOptions options{.channels = {std::nullopt, std::nullopt}};
    sfFDN::FilterBank bypasses(options, designer);
    std::array<float, 8> input = {1.F, 2.F, 3.F, 4.F, -1.F, -2.F, -3.F, -4.F};
    std::array<float, 8> output{};
    sfFDN::AudioBuffer input_buffer(4U, 2U, input);
    sfFDN::AudioBuffer output_buffer(4U, 2U, output);
    bypasses.Process(input_buffer, output_buffer);
    REQUIRE(input == output);
}
