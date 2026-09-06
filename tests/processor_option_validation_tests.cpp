#include <catch2/catch_test_macros.hpp>

#include "sffdn/sffdn.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
constexpr auto kInvalidInterpolation = static_cast<sfFDN::DelayInterpolationType>(255);

sfFDN::ParallelGainsOptions ValidGainsOptions()
{
    return {.mode = sfFDN::ParallelGainsMode::Parallel, .gains = {2.F, -3.F}, .time_varying_config = {}};
}

sfFDN::SchroederAllpassSectionOptions ValidSchroederOptions()
{
    return {.delays = {1.5F, 4.F}, .gains = {0.5F, -0.25F}, .parallel = false};
}

sfFDN::TimeVaryingSchroederAllpassSectionOptions ValidTimeVaryingSchroederOptions()
{
    return {
        .delays = {2.F, 4.F},
        .gains = {0.5F, -0.25F},
        .time_varying_config =
            {
                {.frequency = 0.01F, .amplitude = 0.1F, .initial_phase = 0.F},
                {.frequency = 0.02F, .amplitude = -0.1F, .initial_phase = 1.F},
            },
        .parallel = false,
    };
}
} // namespace

TEST_CASE("Delay options reject nonrepresentable values before construction", "[delay]")
{
    const std::vector<sfFDN::DelayOptions> invalid_options = {
        {.delay = -1.F, .max_delay = 16U, .interp_type = sfFDN::DelayInterpolationType::None, .lfo_config = {}},
        {.delay = 4.F, .max_delay = 3U, .interp_type = sfFDN::DelayInterpolationType::None, .lfo_config = {}},
        {.delay = 4.F, .max_delay = 3U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}},
        {.delay = 4.F, .max_delay = 16U, .interp_type = kInvalidInterpolation, .lfo_config = {}},
        {.delay = 4.F,
         .max_delay = 16U,
         .interp_type = sfFDN::DelayInterpolationType::Linear,
         .lfo_config = sfFDN::ModulationOptions{.frequency = -0.01F, .amplitude = 0.F, .initial_phase = 0.F}},
        {.delay = 4.F,
         .max_delay = 16U,
         .interp_type = sfFDN::DelayInterpolationType::Linear,
         .lfo_config = sfFDN::ModulationOptions{.frequency = 0.F, .amplitude = 0.F, .initial_phase = 1.1F}},
        {.delay = 4.F,
         .max_delay = 4U,
         .interp_type = sfFDN::DelayInterpolationType::Linear,
         .lfo_config = sfFDN::ModulationOptions{.frequency = 0.F, .amplitude = -1.F, .initial_phase = 0.F}},
    };

    for (const auto& options : invalid_options)
    {
        REQUIRE_THROWS_AS(sfFDN::DelayInterp(options), std::invalid_argument);
        REQUIRE_THROWS_AS(sfFDN::DelayTimeVarying(options), std::invalid_argument);
    }

    REQUIRE_NOTHROW(sfFDN::DelayInterp(
        {.delay = 0.F, .max_delay = 2U, .interp_type = sfFDN::DelayInterpolationType::None, .lfo_config = {}}));
    REQUIRE_NOTHROW(sfFDN::DelayInterp(
        {.delay = 0.F, .max_delay = 2U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}}));
    REQUIRE_NOTHROW(sfFDN::DelayInterp(
        {.delay = 4.F, .max_delay = 4U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}}));
    REQUIRE_NOTHROW(sfFDN::DelayInterp(
        {.delay = 0.F, .max_delay = 4U, .interp_type = sfFDN::DelayInterpolationType::Lagrange, .lfo_config = {}}));
}

TEST_CASE("Delay banks validate cardinality and safe tap capacity", "[delay]")
{
    const sfFDN::DelayBankOptions valid_fixed{
        .delays = {1.F, 3.5F}, .block_size = 3U, .interpolation_type = sfFDN::DelayInterpolationType::Linear};
    REQUIRE_NOTHROW(sfFDN::DelayBank(valid_fixed));

    const std::vector<sfFDN::DelayBankOptions> invalid_fixed = {
        {.delays = {-1.F}, .block_size = 1U, .interpolation_type = sfFDN::DelayInterpolationType::None},
        {.delays = {1.F}, .block_size = 1U, .interpolation_type = kInvalidInterpolation},
        {.delays = {1.F},
         .block_size = std::numeric_limits<uint32_t>::max(),
         .interpolation_type = sfFDN::DelayInterpolationType::None},
    };
    for (const auto& options : invalid_fixed)
    {
        REQUIRE_THROWS_AS(sfFDN::DelayBank(options), std::invalid_argument);
    }

    const sfFDN::DelayBankTimeVaryingOptions valid_varying{
        .delays = {3.F, 5.F},
        .max_delay = 8U,
        .interpolation_type = sfFDN::DelayInterpolationType::Linear,
        .time_varying_config =
            {
                {.frequency = 0.F, .amplitude = -1.F, .initial_phase = 0.F},
                {.frequency = 0.F, .amplitude = 1.F, .initial_phase = 1.F},
            },
    };
    REQUIRE_NOTHROW(sfFDN::DelayBankTimeVarying(valid_varying));

    auto bad_cardinality = valid_varying;
    bad_cardinality.time_varying_config.pop_back();
    REQUIRE_THROWS_AS(sfFDN::DelayBankTimeVarying(bad_cardinality), std::invalid_argument);

    auto bad_capacity = valid_varying;
    bad_capacity.max_delay = 5U;
    REQUIRE_THROWS_AS(sfFDN::DelayBankTimeVarying(bad_capacity), std::invalid_argument);

    const sfFDN::DelayBankTimeVaryingOptions bad_allpass_capacity{
        .delays = {4.F},
        .max_delay = 3U,
        .interpolation_type = sfFDN::DelayInterpolationType::Allpass,
        .time_varying_config = {},
    };
    REQUIRE_THROWS_AS(sfFDN::DelayBankTimeVarying(bad_allpass_capacity), std::invalid_argument);
    REQUIRE_NOTHROW(sfFDN::DelayBankTimeVarying({
        .delays = {4.F},
        .max_delay = 4U,
        .interpolation_type = sfFDN::DelayInterpolationType::Allpass,
        .time_varying_config = {},
    }));
}

TEST_CASE("DattorroDelay validates finite controls and preserves supported normalization", "[dattorro]")
{
    auto valid = sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = 3.F,
                         .max_delay = 1U,
                         .interp_type = sfFDN::DelayInterpolationType::Linear,
                         .lfo_config =
                             sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = -1.F, .initial_phase = 0.F}},
        .blend = 1.F,
        .feedforward = -1.F,
        .feedback = 2.F,
    };
    sfFDN::DattorroDelay delay(valid);
    REQUIRE(delay.GetFeedback() < 1.F);
    valid.feedback = -2.F;
    sfFDN::DattorroDelay negative_feedback_delay(valid);
    REQUIRE(negative_feedback_delay.GetFeedback() > -1.F);
    auto overflow = valid;
    overflow.delay_config.delay = std::numeric_limits<float>::max();
    REQUIRE_THROWS_AS(sfFDN::DattorroDelay(overflow), std::invalid_argument);

    valid.delay_config.delay = 1.F;
    REQUIRE_THROWS_AS(sfFDN::DattorroDelay(valid), std::invalid_argument);
    valid.delay_config.delay = 3.F;
    valid.delay_config.lfo_config->amplitude = 1.5F;
    REQUIRE_THROWS_AS(sfFDN::DattorroDelay(valid), std::invalid_argument);
}

TEST_CASE("ParallelGains options accept empty gains and reject invalid modulation", "[parallel_gains]")
{
    REQUIRE_NOTHROW(sfFDN::ParallelGains(
        sfFDN::ParallelGainsOptions{.mode = sfFDN::ParallelGainsMode::Split, .gains = {}, .time_varying_config = {}}));

    auto invalid = ValidGainsOptions();
    invalid.mode = static_cast<sfFDN::ParallelGainsMode>(255);
    REQUIRE_THROWS_AS(sfFDN::ParallelGains(invalid), std::invalid_argument);

    invalid = ValidGainsOptions();
    invalid.time_varying_config = {{.frequency = 0.F, .amplitude = 2.F, .initial_phase = 0.F}};
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingParallelGains(invalid), std::invalid_argument);

    invalid = ValidGainsOptions();
    invalid.time_varying_config = {
        {.frequency = 0.F, .amplitude = 2.F, .initial_phase = 0.F},
        {.frequency = 0.F, .amplitude = -3.F, .initial_phase = 0.F},
    };
    REQUIRE_NOTHROW(sfFDN::TimeVaryingParallelGains(invalid));
}

TEST_CASE("Schroeder sections validate stage vectors before allocation", "[time_varying_allpass]")
{
    auto static_options = ValidSchroederOptions();
    sfFDN::SchroederAllpassSection static_section(static_options);
    REQUIRE(static_section.GetDelays() == std::vector<uint32_t>{1U, 4U});
    REQUIRE_NOTHROW(sfFDN::SchroederAllpassSection({.delays = {0.F}, .gains = {0.F}, .parallel = false}));
    REQUIRE_NOTHROW(sfFDN::SchroederAllpassSection({.delays = {1.F}, .gains = {2.F}, .parallel = false}));
    REQUIRE_THROWS_AS(sfFDN::SchroederAllpassSection({.delays = {}, .gains = {}, .parallel = false}),
                      std::invalid_argument);
    static_options.gains.pop_back();
    REQUIRE_THROWS_AS(sfFDN::SchroederAllpassSection(static_options), std::invalid_argument);
    static_options = ValidSchroederOptions();
    static_options.delays[0] = std::numeric_limits<float>::max();
    REQUIRE_THROWS_AS(sfFDN::SchroederAllpassSection(static_options), std::invalid_argument);

    auto time_varying = ValidTimeVaryingSchroederOptions();
    REQUIRE_NOTHROW(sfFDN::TimeVaryingSchroederAllpassSection(time_varying));
    time_varying.delays[0] = 2.5F;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(time_varying), std::invalid_argument);
    time_varying = ValidTimeVaryingSchroederOptions();
    time_varying.time_varying_config[0].frequency = 0.F;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(time_varying), std::invalid_argument);
    time_varying = ValidTimeVaryingSchroederOptions();
    time_varying.time_varying_config[0].amplitude = 0.F;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(time_varying), std::invalid_argument);
    time_varying = ValidTimeVaryingSchroederOptions();
    time_varying.gains[0] = 0.95F;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(time_varying), std::invalid_argument);
    time_varying = ValidTimeVaryingSchroederOptions();
    time_varying.time_varying_config.pop_back();
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(time_varying), std::invalid_argument);
}
