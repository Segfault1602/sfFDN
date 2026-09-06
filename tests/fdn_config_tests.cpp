#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "sffdn/config_diagnostics.h"
#include "sffdn/sffdn.h"

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
    const auto validation = sfFDN::ValidateFDNStructure(config);

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

    REQUIRE(sfFDN::ValidateFDNStructure(config).has_value());
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(config));
}

TEST_CASE("FDNConfig aggregates independent root issues without dependent noise", "[fdn]")
{
    auto config = MakeValidConfig();
    config.fdn_size = 0;
    config.block_size = 0;
    config.sample_rate = std::numeric_limits<float>::infinity();
    config.direct_gain = std::numeric_limits<float>::quiet_NaN();

    const auto& issues = RequireIssues(sfFDN::ValidateFDNStructure(config));
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
        .time_varying_config = {{.frequency = std::numeric_limits<float>::infinity(),
                                 .amplitude = 0.F,
                                 .initial_phase = 0.F}},
    };

    const auto& issues = RequireIssues(sfFDN::ValidateFDNStructure(config));
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

    const auto first = sfFDN::ValidateFDNStructure(config);
    const auto second = sfFDN::ValidateFDNStructure(config);
    const auto& issues = RequireIssues(first);
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
}

TEST_CASE("FDNConfig reports empty structural arrays without indexing them", "[fdn]")
{
    auto config = MakeValidConfig();
    config.delay_bank_config.delays.clear();
    config.input_block_config.parallel_gains_config.gains.clear();
    config.output_block_config.parallel_gains_config.gains.clear();
    config.attenuation_filter_bank_config = MakeAttenuationBank(0);

    const auto& issues = RequireIssues(sfFDN::ValidateFDNStructure(config));
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
    const auto& custom_issues = RequireIssues(sfFDN::ValidateFDNStructure(custom_matrix));
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
    const auto& hadamard_issues = RequireIssues(sfFDN::ValidateFDNStructure(hadamard));
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

    const auto& issues = RequireIssues(sfFDN::ValidateFDNStructure(config));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::CapacityOverflow, "/fdn_size"));

    auto delay_capacity = MakeValidConfig();
    delay_capacity.delay_bank_config.block_size = std::numeric_limits<uint32_t>::max();
    const auto& delay_issues = RequireIssues(sfFDN::ValidateFDNStructure(delay_capacity));
    REQUIRE(HasIssue(delay_issues, sfFDN::ConfigErrorCode::CapacityOverflow, "/delay_bank_config/block_size"));
}

TEST_CASE("FDNConfig reports invalid single-channel processors at canonical paths", "[fdn]")
{
    const sfFDN::DelayOptions invalid_delay{
        .delay = 4.F, .max_delay = 3U, .interp_type = sfFDN::DelayInterpolationType::Allpass, .lfo_config = {}};

    auto input = MakeValidConfig();
    input.input_block_config.single_channel_processors.emplace_back(invalid_delay);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(input)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/0/DelayOptions/max_delay"));

    auto output = MakeValidConfig();
    output.output_block_config.single_channel_processors.emplace_back(invalid_delay);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(output)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/single_channel_processors/0/DelayOptions/max_delay"));

    auto tone = MakeValidConfig();
    tone.tone_correction_filters.emplace_back(invalid_delay);
    const auto validation = sfFDN::ValidateFDNStructure(tone);
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
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(input)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/multichannel_processors/0/MultichannelProcessorOptions/channels/2/"
                     "DelayOptions/max_delay"));

    auto output = MakeValidConfig();
    output.output_block_config.multichannel_processors.emplace_back(channels);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(output)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/output_block_config/multichannel_processors/0/MultichannelProcessorOptions/channels/2/"
                     "DelayOptions/max_delay"));

    auto loop = MakeValidConfig();
    loop.loop_filter_configs.emplace_back(channels);
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(loop)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/0/MultichannelProcessorOptions/channels/2/DelayOptions/max_delay"));
}

TEST_CASE("FDNConfig reports interpolation enums at their option paths", "[fdn]")
{
    constexpr auto kUnknownInterpolation = static_cast<sfFDN::DelayInterpolationType>(255);

    auto delay = MakeValidConfig();
    delay.input_block_config.single_channel_processors.emplace_back(
        sfFDN::DelayOptions{.delay = 4.F, .max_delay = 8U, .interp_type = kUnknownInterpolation, .lfo_config = {}});
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(delay)), sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/input_block_config/single_channel_processors/0/DelayOptions/interp_type"));

    auto dattorro = MakeValidConfig();
    dattorro.output_block_config.single_channel_processors.emplace_back(sfFDN::DattorroDelayOptions{
        .delay_config = {.delay = 4.F, .max_delay = 8U, .interp_type = kUnknownInterpolation, .lfo_config = {}},
        .blend = 0.F,
        .feedforward = 0.F,
        .feedback = 0.F,
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(dattorro)), sfFDN::ConfigErrorCode::UnsupportedValue,
                     "/output_block_config/single_channel_processors/0/DattorroDelayOptions/delay_config/interp_type"));

    auto bank = MakeValidConfig();
    bank.input_block_config.multichannel_processors.emplace_back(sfFDN::DelayBankOptions{
        .delays = {4.F, 4.F, 4.F, 4.F}, .block_size = 1U, .interpolation_type = kUnknownInterpolation});
    const auto issues = RequireIssues(sfFDN::ValidateFDNStructure(bank));
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
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(dattorro)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/0/DattorroDelayOptions/delay_config/delay"));

    auto static_schroeder = MakeValidConfig();
    static_schroeder.output_block_config.single_channel_processors.emplace_back(
        sfFDN::SchroederAllpassSectionOptions{.delays = {1.F}, .gains = {}, .parallel = false});
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(static_schroeder)), sfFDN::ConfigErrorCode::SizeMismatch,
                     "/output_block_config/single_channel_processors/0/SchroederAllpassSectionOptions/gains"));

    auto time_varying_schroeder = MakeValidConfig();
    time_varying_schroeder.tone_correction_filters.emplace_back(sfFDN::TimeVaryingSchroederAllpassSectionOptions{
        .delays = {1.F},
        .gains = {0.F},
        .time_varying_config = {{.frequency = 0.F, .amplitude = 0.1F, .initial_phase = 0.F}},
        .parallel = false,
    });
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNStructure(time_varying_schroeder)),
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
    const auto input_issues = RequireIssues(sfFDN::ValidateFDNStructure(input));
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
    const auto output_issues = RequireIssues(sfFDN::ValidateFDNStructure(output));
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
    const auto loop_issues = RequireIssues(sfFDN::ValidateFDNStructure(loop));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/0/ParallelGainsConfig/time_varying_config/0/frequency"));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/loop_filter_configs/1/DelayBankTimeVaryingOptions/max_delay"));
    REQUIRE(HasIssue(loop_issues, sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/loop_filter_configs/2/DelayBankOptions/block_size"));
}
