#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "sffdn/config_diagnostics.h"
#include "sffdn/sffdn.h"
#include <sffdn/serialization.h>

namespace
{

sfFDN::FDNConfig MakeValidConfig()
{
    constexpr uint32_t kOrder = 4;
    sfFDN::FDNConfig config{};
    config.fdn_size = kOrder;
    config.block_size = 8;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {8.F, 9.F, 10.F, 11.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Split,
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = kOrder,
        .type = sfFDN::ScalarMatrixType::Hadamard,
    };
    config.output_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Merge,
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    return config;
}

sfFDN::AttenuationFilterBankOptions MakeAttenuationBank(size_t count)
{
    sfFDN::AttenuationFilterBankOptions bank;
    for (size_t index = 0; index < count; ++index)
    {
        bank.filter_configs.emplace_back(
            sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 0.F, .sample_rate = 48000.F});
    }
    return bank;
}

bool HasIssue(const std::vector<sfFDN::ConfigIssue>& issues, sfFDN::ConfigErrorCode code, std::string_view path)
{
    for (const auto& issue : issues)
    {
        if (issue.code == code && issue.path == path)
        {
            return true;
        }
    }
    return false;
}

std::vector<sfFDN::ConfigIssue> RequireIssues(const std::expected<void, std::vector<sfFDN::ConfigIssue>>& validation)
{
    REQUIRE_FALSE(validation.has_value());
    return validation.error();
}

} // namespace

TEST_CASE("FDNConfig validates a usable structural configuration", "[fdn]")
{
    auto config = MakeValidConfig();
    const auto validation = sfFDN::ValidateFDNConfig(config);

    REQUIRE(validation.has_value());
    const auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn != nullptr);
}

TEST_CASE("FDNConfig accepts short inserts and shared attenuation", "[fdn]")
{
    auto config = MakeValidConfig();
    config.input_block_config.multichannel_processors.emplace_back(sfFDN::DelayBankOptions{
        .delays = {1.F, 2.F, 3.F, 4.F},
        .block_size = 4U,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    });
    config.attenuation_filter_bank_config = MakeAttenuationBank(1);
    config.loop_filter_configs.emplace_back(MakeAttenuationBank(1));

    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(config));
}

TEST_CASE("FDNConfig aggregates independent root issues without dependent noise", "[fdn]")
{
    auto config = MakeValidConfig();
    config.fdn_size = 0;
    config.block_size = 0;
    config.sample_rate = std::numeric_limits<float>::infinity();
    config.direct_gain = std::numeric_limits<float>::quiet_NaN();

    const auto& issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE(issues.size() == 4U);
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/fdn_size"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/block_size"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/sample_rate"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/direct_gain"));
}

TEST_CASE("FDNConfig reports indexed nested structural issues", "[fdn]")
{
    auto config = MakeValidConfig();
    config.input_block_config.parallel_gains_config.mode = sfFDN::ParallelGainsMode::Parallel;
    config.input_block_config.parallel_gains_config.gains.pop_back();
    config.input_block_config.multichannel_processors.emplace_back(MakeAttenuationBank(2));
    config.loop_filter_configs.emplace_back(sfFDN::MultichannelProcessorOptions{.channels = {std::nullopt}});
    config.feedback_matrix_config = sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = 4,
        .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
        .time_varying_config = {{.frequency = -0.01F,
                                 .amplitude = 0.F,
                                 .initial_phase = 0.F}},
    };

    const auto& issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE(
        HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue, "/input_block_config/parallel_gains_config/mode"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/input_block_config/parallel_gains_config/gains"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/input_block_config/multichannel_processors/0/AttenuationFilterBankOptions"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/loop_filter_configs/0/MultichannelProcessorOptions/channels"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/feedback_matrix_config/TimeVaryingFeedbackMatrixOptions/time_varying_config"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/TimeVaryingFeedbackMatrixOptions/time_varying_config/0/frequency"));
}

TEST_CASE("FDNConfig reports deterministic unchanged validation and typed factory errors", "[fdn]")
{
    auto config = MakeValidConfig();
    config.delay_bank_config.delays[0] = -1.F;
    config.output_block_config.parallel_gains_config.gains.pop_back();

    const auto first = sfFDN::ValidateFDNConfig(config);
    const auto second = sfFDN::ValidateFDNConfig(config);
    const auto issues = RequireIssues(first);
    REQUIRE(issues == RequireIssues(second));
    REQUIRE(config.delay_bank_config.delays[0] == -1.F);
    REQUIRE(config.delay_bank_config.delays.size() == 4U);
    REQUIRE(config.output_block_config.parallel_gains_config.gains.size() == 3U);

    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(config), std::runtime_error);
    try
    {
        static_cast<void>(sfFDN::CreateFDNFromConfig(config));
        FAIL("CreateFDNFromConfig must reject structural errors");
    }
    catch (const sfFDN::FDNConfigError& error)
    {
        REQUIRE(error.Issues() == issues);
        const std::string_view message(error.what());
        for (const auto& issue : issues)
        {
            REQUIRE(message.find(issue.path) != std::string_view::npos);
            REQUIRE(message.find(issue.message) != std::string_view::npos);
        }
    }

    config.delay_bank_config.delays[0] = 8.F;
    config.output_block_config.parallel_gains_config.gains.push_back(1.F);
    REQUIRE(first.error() == issues);
}

TEST_CASE("FDNConfig reports empty structural arrays without indexing them", "[fdn]")
{
    auto config = MakeValidConfig();
    config.delay_bank_config.delays.clear();
    config.input_block_config.parallel_gains_config.gains.clear();
    config.output_block_config.parallel_gains_config.gains.clear();
    config.attenuation_filter_bank_config = MakeAttenuationBank(0);

    const auto& issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/delay_bank_config/delays"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/input_block_config/parallel_gains_config/gains"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/output_block_config/parallel_gains_config/gains"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/attenuation_filter_bank_config/AttenuationFilterBankOptions"));
}

TEST_CASE("FDNConfig reports feedback matrix dimensions", "[fdn]")
{
    auto custom_matrix = MakeValidConfig();
    custom_matrix.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = custom_matrix.fdn_size,
        .custom_matrix = std::vector<float>(3, 0.F),
    };
    const auto& custom_issues = RequireIssues(sfFDN::ValidateFDNConfig(custom_matrix));
    REQUIRE(HasIssue(custom_issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/feedback_matrix_config/ScalarFeedbackMatrixOptions/custom_matrix"));

    auto hadamard = MakeValidConfig();
    hadamard.fdn_size = 3;
    hadamard.delay_bank_config.delays.pop_back();
    hadamard.input_block_config.parallel_gains_config.gains.pop_back();
    hadamard.output_block_config.parallel_gains_config.gains.pop_back();
    hadamard.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = hadamard.fdn_size,
        .type = sfFDN::ScalarMatrixType::Hadamard,
    };
    const auto& hadamard_issues = RequireIssues(sfFDN::ValidateFDNConfig(hadamard));
    REQUIRE(HasIssue(hadamard_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/ScalarFeedbackMatrixOptions/matrix_size"));
}

TEST_CASE("FDNConfig reports capacity overflow without constructing", "[fdn]")
{
    auto config = MakeValidConfig();
    config.block_size = 1073741824U;
    config.delay_bank_config = {
        .delays = {1073741824.F, 1073741824.F, 1073741824.F, 1073741824.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };

    const auto& issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::CapacityOverflow, "/fdn_size"));

    auto delay_capacity = MakeValidConfig();
    delay_capacity.delay_bank_config.block_size = std::numeric_limits<uint32_t>::max();
    const auto& delay_issues = RequireIssues(sfFDN::ValidateFDNConfig(delay_capacity));
    REQUIRE(HasIssue(delay_issues, sfFDN::ConfigErrorCode::CapacityOverflow, "/delay_bank_config/block_size"));
}

TEST_CASE("FDNConfig reports invalid single-channel processors at canonical paths", "[fdn]")
{
    const sfFDN::DelayOptions invalid_delay{
        .delay = 4.F, .max_delay = 3U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}};

    auto input = MakeValidConfig();
    input.input_block_config.single_channel_processors.emplace_back(invalid_delay);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(input)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/0/DelayOptions/max_delay"));

    auto output = MakeValidConfig();
    output.output_block_config.single_channel_processors.emplace_back(invalid_delay);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(output)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/single_channel_processors/0/DelayOptions/max_delay"));

    auto tone = MakeValidConfig();
    tone.tone_correction_filters.emplace_back(invalid_delay);
    const auto validation = sfFDN::ValidateFDNConfig(tone);
    const auto issues = RequireIssues(validation);
    REQUIRE(
        HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/tone_correction_filters/0/DelayOptions/max_delay"));
    REQUIRE_THROWS_AS(sfFDN::DelayInterp(invalid_delay), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(tone), sfFDN::FDNConfigError);
}

TEST_CASE("FDNConfig validates multichannel delays without indexing missing channels", "[fdn]")
{
    const sfFDN::DelayOptions valid_delay{
        .delay = 4.F, .max_delay = 8U, .interp_type = sfFDN::DelayInterpolationType::None, .lfo_config = {}};
    const sfFDN::DelayOptions invalid_delay{
        .delay = 4.F, .max_delay = 3U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}};

    sfFDN::MultichannelProcessorOptions channels;
    channels.channels.emplace_back(std::nullopt);
    channels.channels.emplace_back(valid_delay);
    channels.channels.emplace_back(invalid_delay);
    channels.channels.emplace_back(std::nullopt);

    auto input = MakeValidConfig();
    input.input_block_config.multichannel_processors.emplace_back(channels);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(input)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/multichannel_processors/0/MultichannelProcessorOptions/channels/2/"
                     "DelayOptions/max_delay"));

    auto output = MakeValidConfig();
    output.output_block_config.multichannel_processors.emplace_back(channels);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(output)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/multichannel_processors/0/MultichannelProcessorOptions/channels/2/"
                     "DelayOptions/max_delay"));

    auto loop = MakeValidConfig();
    loop.loop_filter_configs.emplace_back(channels);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(loop)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/0/MultichannelProcessorOptions/channels/2/DelayOptions/max_delay"));
}

TEST_CASE("FDNConfig reports interpolation enums at their option paths", "[fdn]")
{
    constexpr auto kUnknownInterpolation = static_cast<sfFDN::DelayInterpolationType>(255);

    auto delay = MakeValidConfig();
    delay.input_block_config.single_channel_processors.emplace_back(
        sfFDN::DelayOptions{.delay = 4.F, .max_delay = 8U, .interp_type = kUnknownInterpolation, .lfo_config = {}});
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(delay)), sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/input_block_config/single_channel_processors/0/DelayOptions/interp_type"));

    auto dattorro = MakeValidConfig();
    dattorro.output_block_config.single_channel_processors.emplace_back(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = 4.F, .max_delay = 8U, .interp_type = kUnknownInterpolation, .lfo_config = {}},
        .blend = 0.F,
        .feedforward = 0.F,
        .feedback = 0.F,
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(dattorro)), sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/output_block_config/single_channel_processors/0/DattorroDelayOptions/delay_config/interp_type"));

    auto bank = MakeValidConfig();
    bank.input_block_config.multichannel_processors.emplace_back(sfFDN::DelayBankOptions{
        .delays = {4.F, 4.F, 4.F, 4.F}, .block_size = 1U, .interpolation_type = kUnknownInterpolation});
    const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(bank));
    REQUIRE(issues.size() == 1U);
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/input_block_config/multichannel_processors/0/DelayBankOptions/interpolation_type"));
}

TEST_CASE("FDNConfig reports processor option errors from each single-channel variant", "[fdn]")
{
    auto dattorro = MakeValidConfig();
    dattorro.input_block_config.single_channel_processors.emplace_back(sfFDN::DattorroDelayOptions{
        .delay_config =
            {.delay = 1.F, .max_delay = 4U, .interp_type = sfFDN::DelayInterpolationType::None, .lfo_config = {}},
        .blend = 0.F,
        .feedforward = 0.F,
        .feedback = 0.F,
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(dattorro)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/0/DattorroDelayOptions/delay_config/delay"));

    auto static_schroeder = MakeValidConfig();
    static_schroeder.output_block_config.single_channel_processors.emplace_back(
        sfFDN::SchroederAllpassSectionOptions{.delays = {1.F}, .gains = {}, .parallel = false});
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(static_schroeder)), sfFDN::ConfigErrorCode::SizeMismatch,
                     "/output_block_config/single_channel_processors/0/SchroederAllpassSectionOptions/gains"));

    auto time_varying_schroeder = MakeValidConfig();
    time_varying_schroeder.tone_correction_filters.emplace_back(sfFDN::TimeVaryingSchroederAllpassSectionOptions{
        .delays = {1.F},
        .gains = {0.F},
        .time_varying_config = {{.frequency = 0.F, .amplitude = 0.1F, .initial_phase = 0.F}},
        .parallel = false,
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(time_varying_schroeder)),
                     sfFDN::ConfigErrorCode::InvalidValue,
                     "/tone_correction_filters/0/TimeVaryingSchroederAllpassSectionOptions/time_varying_config/0/"
                     "frequency"));
}

TEST_CASE("FDNConfig validates time-varying gains and delay banks in each multichannel placement", "[fdn]")
{
    const sfFDN::ParallelGainsOptions invalid_gains{
        .mode = sfFDN::ParallelGainsMode::Parallel,
        .gains = {1.F, 1.F, 1.F, 1.F},
        .time_varying_config = {{.frequency = -0.01F, .amplitude = 2.F, .initial_phase = 0.F},
                                {.frequency = 0.F, .amplitude = 2.F, .initial_phase = 0.F},
                                {.frequency = 0.F, .amplitude = 2.F, .initial_phase = 0.F},
                                {.frequency = 0.F, .amplitude = 2.F, .initial_phase = 0.F}},
    };
    const sfFDN::DelayBankTimeVaryingOptions invalid_bank{
        .delays = {4.F, 4.F, 4.F, 4.F},
        .max_delay = 3U,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
        .time_varying_config = {},
    };
    const sfFDN::DelayBankOptions invalid_fixed_bank{
        .delays = {4.F, 4.F, 4.F, 4.F},
        .block_size = std::numeric_limits<uint32_t>::max(),
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };

    auto input = MakeValidConfig();
    input.input_block_config.parallel_gains_config = invalid_gains;
    input.input_block_config.multichannel_processors.emplace_back(invalid_bank);
    input.input_block_config.multichannel_processors.emplace_back(invalid_gains);
    input.input_block_config.multichannel_processors.emplace_back(invalid_fixed_bank);
    const auto input_issues = RequireIssues(sfFDN::ValidateFDNConfig(input));
    REQUIRE(HasIssue(input_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/parallel_gains_config/time_varying_config/0/frequency"));
    REQUIRE(HasIssue(input_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/multichannel_processors/0/DelayBankTimeVaryingOptions/max_delay"));
    REQUIRE(HasIssue(input_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/multichannel_processors/1/ParallelGainsConfig/time_varying_config/0/"
                     "frequency"));
    REQUIRE(HasIssue(input_issues, sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/input_block_config/multichannel_processors/2/DelayBankOptions/block_size"));

    auto output = MakeValidConfig();
    output.output_block_config.parallel_gains_config = invalid_gains;
    output.output_block_config.multichannel_processors.emplace_back(invalid_bank);
    output.output_block_config.multichannel_processors.emplace_back(invalid_fixed_bank);
    output.output_block_config.multichannel_processors.emplace_back(invalid_gains);
    const auto output_issues = RequireIssues(sfFDN::ValidateFDNConfig(output));
    REQUIRE(HasIssue(output_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/parallel_gains_config/time_varying_config/0/frequency"));
    REQUIRE(HasIssue(output_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/multichannel_processors/0/DelayBankTimeVaryingOptions/max_delay"));
    REQUIRE(HasIssue(output_issues, sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/output_block_config/multichannel_processors/1/DelayBankOptions/block_size"));
    REQUIRE(HasIssue(output_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/multichannel_processors/2/ParallelGainsConfig/time_varying_config/0/"
                     "frequency"));

    auto loop = MakeValidConfig();
    loop.loop_filter_configs.emplace_back(invalid_gains);
    loop.loop_filter_configs.emplace_back(invalid_bank);
    loop.loop_filter_configs.emplace_back(invalid_fixed_bank);
    const auto loop_issues = RequireIssues(sfFDN::ValidateFDNConfig(loop));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/0/ParallelGainsConfig/time_varying_config/0/frequency"));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/1/DelayBankTimeVaryingOptions/max_delay"));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/loop_filter_configs/2/DelayBankOptions/block_size"));
}

TEST_CASE("FDNConfig validates supported filter and nonlinear processor domains", "[fdn]")
{
    auto config = MakeValidConfig();
    config.input_block_config.single_channel_processors = {
        sfFDN::AllpassFilterOptions{.coeff = 2.F},
        sfFDN::CascadedBiquadsOptions{.coeffs = {{1.F, 0.F, 0.F, -1.F, 2.F, 0.F}}},
        sfFDN::FirOptions{.coeffs = {0.F, 0.F}},
        sfFDN::GraphicEQOptions{
            .gains_db = {},
            .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
            .sample_rate = 48000.F},
    };
    config.input_block_config.multichannel_processors.emplace_back(sfFDN::MultichannelProcessorOptions{
        .channels = {sfFDN::AllpassFilterOptions{.coeff = -2.F}, std::nullopt, sfFDN::FirOptions{.coeffs = {0.F}},
                     sfFDN::RingModulatorOptions{.frequency = 0.F, .amplitude = -2.F, .initial_phase = 1.F}},
    });
    config.output_block_config.multichannel_processors.emplace_back(sfFDN::MultichannelProcessorOptions{
        .channels = {std::nullopt, sfFDN::CascadedBiquadsOptions{.coeffs = {}}, std::nullopt,
                     sfFDN::GraphicEQOptions{
                         .gains_db = {},
                         .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
                         .sample_rate = 48000.F}},
    });
    config.output_block_config.single_channel_processors = {
        sfFDN::ControllableFullWaveRectifierOptions{.alpha = 1.F, .dc_block = false, .sample_rate = 0.F},
        sfFDN::SignalDependentFractionalDelayOptions{.d = 1.F},
        sfFDN::RingModulatorOptions{.frequency = 0.F, .amplitude = -1.F, .initial_phase = 0.F},
    };
    config.tone_correction_filters = {sfFDN::CascadedBiquadsOptions{.coeffs = {}}};

    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto invalid_iir = config;
    std::get<sfFDN::CascadedBiquadsOptions>(invalid_iir.input_block_config.single_channel_processors[1]).coeffs[0].a0 =
        0.F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_iir)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/1/CascadedBiquadsOptions/coeffs/0/a0"));

    auto invalid_fir = config;
    std::get<sfFDN::FirOptions>(invalid_fir.input_block_config.single_channel_processors[2]).coeffs.clear();
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_fir)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/2/FirOptions/coeffs"));

    auto invalid_graphic = config;
    std::get<sfFDN::GraphicEQOptions>(invalid_graphic.input_block_config.single_channel_processors[3]).freqs[3] = 125.F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_graphic)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/3/GraphicEQOptions/freqs/3"));

    auto invalid_rectifier = config;
    std::get<sfFDN::ControllableFullWaveRectifierOptions>(
        invalid_rectifier.output_block_config.single_channel_processors[0])
        .alpha = -0.1F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_rectifier)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/single_channel_processors/0/ControllableFullWaveRectifierOptions/alpha"));

    auto invalid_fractional_delay = config;
    std::get<sfFDN::SignalDependentFractionalDelayOptions>(
        invalid_fractional_delay.output_block_config.single_channel_processors[1])
        .d = 1.1F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_fractional_delay)),
                     sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/single_channel_processors/1/SignalDependentFractionalDelayOptions/d"));

    auto invalid_ring = config;
    std::get<sfFDN::RingModulatorOptions>(invalid_ring.output_block_config.single_channel_processors[2]).frequency = -1.F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_ring)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/single_channel_processors/2/RingModulatorOptions/frequency"));
}

TEST_CASE("FDNConfig validates attenuation and matrix domains at legacy option paths", "[fdn]")
{
    auto config = MakeValidConfig();
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = 4,
        .type = sfFDN::ScalarMatrixType::Count,
        .custom_matrix = std::vector<float>{2.F, 0.F, 0.F, 0.F, 0.F, 2.F, 0.F, 0.F,
                                            0.F, 0.F, 2.F, 0.F, 0.F, 0.F, 0.F, 2.F},
    };
    config.attenuation_filter_bank_config = sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {
            sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 0.F, .sample_rate = 48000.F},
            sfFDN::TwoBandFilterOptions{.t60s = {1.F, 0.5F}, .delay = -1.F, .sample_rate = 48000.F},
            sfFDN::ThreeBandFilterOptions{
                .t60s = {1.F, 0.8F, 0.5F}, .delay = 4.F, .freqs = {800.F, 8000.F}, .q = 1.F, .sample_rate = 48000.F},
            sfFDN::TenBandFilterOptions{
                .t60s = {1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F},
                .delay = 4.F,
                .sample_rate = 48000.F,
                .shelf_cutoff = 8000.F},
        },
    };
    config.loop_filter_configs.emplace_back(*config.attenuation_filter_bank_config);
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto invalid_insert = config;
    invalid_insert.input_block_config.multichannel_processors.emplace_back(sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = -1.F, .sample_rate = 48000.F}},
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_insert)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/multichannel_processors/0/AttenuationFilterBankOptions/0/"
                     "ProportionalAttenuationConfig/delay"));
    const nlohmann::json invalid_insert_json = invalid_insert;
    REQUIRE_NOTHROW(invalid_insert_json.at(nlohmann::json::json_pointer(
        "/input_block_config/multichannel_processors/0/AttenuationFilterBankOptions/0/"
        "ProportionalAttenuationConfig/delay")));

    auto invalid_output_insert = config;
    invalid_output_insert.output_block_config.multichannel_processors.emplace_back(sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = -1.F, .sample_rate = 48000.F}},
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_output_insert)),
                     sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/multichannel_processors/0/AttenuationFilterBankOptions/0/"
                     "ProportionalAttenuationConfig/delay"));

    auto invalid_matrix = config;
    invalid_matrix.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4, .stage_count = 1, .sparsity = 0.F, .type = sfFDN::ScalarMatrixType::Count};
    const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(invalid_matrix));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/sparsity"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/type"));

    auto invalid_time_varying = config;
    invalid_time_varying.feedback_matrix_config = sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = 4,
        .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
        .time_varying_config = {{.frequency = 0.F, .amplitude = 1.1F, .initial_phase = 0.F}},
    };
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_time_varying)), sfFDN::ConfigErrorCode::SizeMismatch,
                     "/feedback_matrix_config/TimeVaryingFeedbackMatrixOptions/time_varying_config"));
}

TEST_CASE("FDNConfig accepts supported matrix types and rejects matrix and attenuation boundaries", "[fdn]")
{
    constexpr std::array kMatrixTypes = {
        sfFDN::ScalarMatrixType::Identity,          sfFDN::ScalarMatrixType::Random,
        sfFDN::ScalarMatrixType::Householder,       sfFDN::ScalarMatrixType::RandomHouseholder,
        sfFDN::ScalarMatrixType::Hadamard,          sfFDN::ScalarMatrixType::Circulant,
        sfFDN::ScalarMatrixType::Allpass,           sfFDN::ScalarMatrixType::NestedAllpass,
        sfFDN::ScalarMatrixType::VariableDiffusion,
    };
    for (const auto type : kMatrixTypes)
    {
        auto config = MakeValidConfig();
        config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
            .matrix_size = config.fdn_size, .type = type, .rng_seed = 42U, .arg = 0.5F};
        REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
    }

    auto invalid_scalar = MakeValidConfig();
    invalid_scalar.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = 4, .type = sfFDN::ScalarMatrixType::VariableDiffusion, .arg = 2.F};
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(invalid_scalar)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/ScalarFeedbackMatrixOptions/arg"));

    auto valid_cascade = MakeValidConfig();
    valid_cascade.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = valid_cascade.fdn_size,
        .stage_count = 0U,
        .sparsity = 1.F,
        .type = sfFDN::ScalarMatrixType::Random};
    REQUIRE(sfFDN::ValidateFDNConfig(valid_cascade).has_value());

    auto overflow_cascade = valid_cascade;
    std::get<sfFDN::CascadedFeedbackMatrixOptions>(overflow_cascade.feedback_matrix_config).stage_count =
        std::numeric_limits<uint32_t>::max();
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(overflow_cascade)),
                     sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/stage_count"));

    auto shift_overflow = MakeValidConfig();
    shift_overflow.fdn_size = 64U;
    shift_overflow.delay_bank_config.delays.assign(64U, 8.F);
    shift_overflow.input_block_config.parallel_gains_config.gains.assign(64U, 1.F);
    shift_overflow.output_block_config.parallel_gains_config.gains.assign(64U, 1.F);
    shift_overflow.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 64U, .stage_count = 6U, .sparsity = 1.F, .type = sfFDN::ScalarMatrixType::Random};
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(shift_overflow)),
                     sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/stage_count"));

    auto shift_neighbor = shift_overflow;
    shift_neighbor.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 64U, .stage_count = 5U, .sparsity = 1.F, .type = sfFDN::ScalarMatrixType::Random};
    REQUIRE(sfFDN::ValidateFDNConfig(shift_neighbor).has_value());

    auto gain_overflow = MakeValidConfig();
    gain_overflow.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4U,
        .stage_count = 2U,
        .sparsity = 3.F,
        .type = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 100.F};
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(gain_overflow)),
                     sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/gain_per_samples"));

    auto negative_gain_overflow = gain_overflow;
    std::get<sfFDN::CascadedFeedbackMatrixOptions>(negative_gain_overflow.feedback_matrix_config).gain_per_samples =
        -100.F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(negative_gain_overflow)),
                     sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/gain_per_samples"));

    auto gain_neighbor = MakeValidConfig();
    gain_neighbor.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4U,
        .stage_count = 2U,
        .sparsity = 3.F,
        .type = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 1.F};
    REQUIRE(sfFDN::ValidateFDNConfig(gain_neighbor).has_value());
    auto& gain_neighbor_options =
        std::get<sfFDN::CascadedFeedbackMatrixOptions>(gain_neighbor.feedback_matrix_config);
    gain_neighbor_options.gain_per_samples = -1.F;
    REQUIRE(sfFDN::ValidateFDNConfig(gain_neighbor).has_value());

    auto fractional_negative_gain = MakeValidConfig();
    fractional_negative_gain.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4U,
        .stage_count = 2U,
        .sparsity = 1.1F,
        .type = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = -0.5F};
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(fractional_negative_gain)),
                     sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/gain_per_samples"));
    auto& fractional_negative_gain_options =
        std::get<sfFDN::CascadedFeedbackMatrixOptions>(fractional_negative_gain.feedback_matrix_config);
    fractional_negative_gain_options.gain_per_samples = 0.5F;
    REQUIRE(sfFDN::ValidateFDNConfig(fractional_negative_gain).has_value());

    auto integral_negative_gain = gain_neighbor;
    auto& integral_negative_gain_options =
        std::get<sfFDN::CascadedFeedbackMatrixOptions>(integral_negative_gain.feedback_matrix_config);
    integral_negative_gain_options.sparsity = 3.F;
    integral_negative_gain_options.gain_per_samples = -0.5F;
    REQUIRE(sfFDN::ValidateFDNConfig(integral_negative_gain).has_value());

    auto normalization_overflow = MakeValidConfig();
    normalization_overflow.tone_correction_filters = {sfFDN::CascadedBiquadsOptions{
        .coeffs = {{std::numeric_limits<float>::max(), 0.F, 0.F, std::numeric_limits<float>::min(), 0.F, 0.F}}}};
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(normalization_overflow)),
                     sfFDN::ConfigErrorCode::InvalidValue,
                     "/tone_correction_filters/0/CascadedBiquadsOptions/coeffs/0/b0"));

    auto invalid_three_band = MakeValidConfig();
    invalid_three_band.attenuation_filter_bank_config = sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {sfFDN::ThreeBandFilterOptions{
            .t60s = {1.F, 1.F, 1.F}, .delay = 4.F, .freqs = {8000.F, 800.F}, .q = 0.F, .sample_rate = 48000.F}},
    };
    const auto three_band_issues = RequireIssues(sfFDN::ValidateFDNConfig(invalid_three_band));
    REQUIRE(HasIssue(three_band_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/attenuation_filter_bank_config/AttenuationFilterBankOptions/0/ThreeBandFilterConfig/freqs"));
    REQUIRE(HasIssue(three_band_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/attenuation_filter_bank_config/AttenuationFilterBankOptions/0/ThreeBandFilterConfig/q"));

    auto invalid_ten_band = MakeValidConfig();
    invalid_ten_band.attenuation_filter_bank_config = sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {sfFDN::TenBandFilterOptions{
            .t60s = {1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F},
            .delay = 4.F,
            .sample_rate = 32000.F,
            .shelf_cutoff = 16000.F}},
    };
    const auto ten_band_issues = RequireIssues(sfFDN::ValidateFDNConfig(invalid_ten_band));
    REQUIRE(HasIssue(ten_band_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/attenuation_filter_bank_config/AttenuationFilterBankOptions/0/TenBandFilterConfig/sample_rate"));
    REQUIRE(HasIssue(ten_band_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/attenuation_filter_bank_config/AttenuationFilterBankOptions/0/TenBandFilterConfig/shelf_cutoff"));
}
