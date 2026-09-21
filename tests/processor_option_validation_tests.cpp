#include <catch2/catch_test_macros.hpp>

#include "processor_option_validation.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
constexpr auto kInvalidInterpolation = static_cast<sfFDN::DelayInterpolationType>(255);

template <class Options>
concept CanRequireValidOptions =
    requires(Options&& options) { sfFDN::detail::RequireValidOptions(std::forward<Options>(options)); };

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

template <typename Options>
void RequireInvalidOptions(const Options& options)
{
    std::vector<sfFDN::ConfigIssue> issues;
    sfFDN::detail::ValidateOptions(options, "", issues);
    REQUIRE_FALSE(issues.empty());
}

void RequireInvalidAttenuationOptions(const sfFDN::attenuation_filter_variant_t& options)
{
    std::vector<sfFDN::ConfigIssue> issues;
    sfFDN::detail::ValidateAttenuationOptions(options, "", issues, false);
    REQUIRE_FALSE(issues.empty());
}
} // namespace

TEST_CASE("RequireValidOptions borrows only lvalues and preserves their contents", "[fdn_config]")
{
    using Options = sfFDN::DelayBankOptions;
    static_assert(CanRequireValidOptions<Options&>);
    static_assert(CanRequireValidOptions<const Options&>);
    static_assert(!CanRequireValidOptions<Options>);
    static_assert(!CanRequireValidOptions<const Options>);

    Options options{.delays = {4.F, 8.F}, .block_size = 4U};
    const auto original = options;
    REQUIRE(&sfFDN::detail::RequireValidOptions(options) == &options);
    REQUIRE(&sfFDN::detail::RequireValidOptions(original) == &original);
    REQUIRE(options == original);

    options.delays[0] = -1.F;
    const auto invalid = options;
    REQUIRE_THROWS_AS(sfFDN::detail::RequireValidOptions(options), std::invalid_argument);
    REQUIRE(options == invalid);
}

TEST_CASE("ChannelMatrixOptions validates dimensions and coefficient count", "[fdn_config]")
{
    std::vector<sfFDN::ConfigIssue> issues;
    sfFDN::detail::ValidateOptions(
        sfFDN::ChannelMatrixOptions{.input_channel_count = 2U, .output_channel_count = 3U, .coefficients = {1.F}},
        "/matrix", issues);
    REQUIRE(issues.size() == 1U);
    REQUIRE(issues[0].code == sfFDN::ConfigErrorCode::SizeMismatch);
    REQUIRE(issues[0].path == "/matrix/coefficients");

    issues.clear();
    sfFDN::detail::ValidateOptions(
        sfFDN::ChannelMatrixOptions{.input_channel_count = 0U, .output_channel_count = 0U, .coefficients = {}},
        "/matrix", issues);
    REQUIRE(issues.size() == 2U);
    REQUIRE(issues[0].path == "/matrix/input_channel_count");
    REQUIRE(issues[1].path == "/matrix/output_channel_count");
}

TEST_CASE("Kronecker matrix options report nested cardinality and modulation issues", "[fdn_config]")
{
    const sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions options{
        .matrix = {.matrix_size = 8U, .angles = {0.1F}, .kernel_types = {sfFDN::KroneckerKernelType::Rotation}},
        .time_varying_config = {{.frequency = -0.1F, .amplitude = 1.1F, .initial_phase = 1.1F}},
    };
    std::vector<sfFDN::ConfigIssue> issues;
    sfFDN::detail::ValidateOptions(options, "/dynamic", issues);

    REQUIRE(issues.size() == 6U);
    REQUIRE(issues[0].path == "/dynamic/matrix/angles");
    REQUIRE(issues[1].path == "/dynamic/matrix/kernel_types");
    REQUIRE(issues[2].path == "/dynamic/time_varying_config");
    REQUIRE(issues[3].path == "/dynamic/time_varying_config/0/frequency");
    REQUIRE(issues[4].path == "/dynamic/time_varying_config/0/initial_phase");
    REQUIRE(issues[5].path == "/dynamic/time_varying_config/0/amplitude");
}

TEST_CASE("ValidateOptions rejects invalid filter nonlinear and attenuation domains", "[fdn_config]")
{
    RequireInvalidOptions(sfFDN::CascadedBiquadsOptions{
        .coeffs = {{1.F, 0.F, 0.F, 0.F, 2.F, 0.F}},
    });
    RequireInvalidOptions(sfFDN::CascadedBiquadsOptions{
        .coeffs = {{std::numeric_limits<float>::max(), 0.F, 0.F, std::numeric_limits<float>::min(), 0.F, 0.F}},
    });
    RequireInvalidOptions(sfFDN::FirOptions{.coeffs = {}});
    RequireInvalidOptions(sfFDN::GraphicEQOptions{
        .freqs = {32.F, 64.F, 125.F, 125.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
    });
    RequireInvalidOptions(sfFDN::ControllableFullWaveRectifierOptions{.alpha = -0.1F});
    RequireInvalidOptions(sfFDN::SignalDependentFractionalDelayOptions{.d = 1.1F});
    RequireInvalidOptions(sfFDN::RingModulatorOptions{.frequency = -1.F});
    RequireInvalidOptions(sfFDN::VariableDiffusionOptions{.diffusion = 2.F});

    RequireInvalidAttenuationOptions(sfFDN::TwoBandFilterOptions{
        .t60s = {1.F, 0.5F},
        .delay = -1.F,
        .sample_rate = 48000.F,
    });
    RequireInvalidAttenuationOptions(sfFDN::ThreeBandFilterOptions{
        .t60s = {1.F, 1.F, 1.F},
        .delay = 4.F,
        .freqs = {8000.F, 800.F},
        .q = 0.F,
        .sample_rate = 48000.F,
    });
    RequireInvalidAttenuationOptions(sfFDN::TenBandFilterOptions{
        .t60s = {1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F},
        .delay = 4.F,
        .sample_rate = 32000.F,
        .shelf_cutoff = 16000.F,
    });
}

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

TEST_CASE("CascadedBiquads SetCoefficients retains state on valid updates and failures", "[filter]")
{
    const sfFDN::FilterCoefficients old_coefficients{1.F, 0.F, 0.F, 1.F, -0.5F, 0.F};
    const sfFDN::FilterCoefficients updated_coefficients{2.F, 0.F, 0.F, 1.F, -0.5F, 0.F};
    const std::array updated_set = {updated_coefficients};
    const std::array invalid_set = {sfFDN::FilterCoefficients{1.F, 0.F, 0.F, 0.F, 0.F, 0.F}};
    sfFDN::CascadedBiquads filter({.coeffs = {old_coefficients}});

    std::array<float, 1> impulse = {1.F};
    std::array<float, 1> warm_output{};
    sfFDN::AudioBuffer impulse_buffer(impulse);
    sfFDN::AudioBuffer warm_output_buffer(warm_output);
    filter.Process(impulse_buffer, warm_output_buffer);
    auto before_invalid_update = filter.Clone();

    std::array<float, 1> silence = {0.F};
    std::array<float, 1> after_invalid_output{};
    std::array<float, 1> control_output{};
    sfFDN::AudioBuffer silence_buffer(silence);
    sfFDN::AudioBuffer after_invalid_output_buffer(after_invalid_output);
    sfFDN::AudioBuffer control_output_buffer(control_output);
    REQUIRE_THROWS_AS(filter.SetCoefficients(invalid_set), std::invalid_argument);
    filter.Process(silence_buffer, after_invalid_output_buffer);
    before_invalid_update->Process(silence_buffer, control_output_buffer);
    REQUIRE(after_invalid_output == control_output);

    std::array<float, 1> coefficient_control_output{};
    sfFDN::AudioBuffer coefficient_control_output_buffer(coefficient_control_output);
    filter.Process(impulse_buffer, warm_output_buffer);
    before_invalid_update->Process(impulse_buffer, coefficient_control_output_buffer);
    REQUIRE(warm_output == coefficient_control_output);

    filter.SetCoefficients(updated_set);
    std::array<float, 1> retained_output{};
    std::array<float, 1> fresh_output{};
    sfFDN::AudioBuffer retained_output_buffer(retained_output);
    sfFDN::AudioBuffer fresh_output_buffer(fresh_output);
    filter.Process(silence_buffer, retained_output_buffer);
    sfFDN::CascadedBiquads fresh({.coeffs = {updated_coefficients}});
    fresh.Process(silence_buffer, fresh_output_buffer);
    REQUIRE(retained_output[0] == 0.625F);
    REQUIRE(fresh_output[0] == 0.F);
}
