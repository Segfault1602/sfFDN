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
        REQUIRE(std::string_view(error.what()).find("/delay_bank_config/delays/0") != std::string_view::npos);
        REQUIRE(std::string_view(error.what()).find("/output_block_config/parallel_gains_config/gains") !=
                std::string_view::npos);
        REQUIRE(std::string_view(error.what()).find("delay must be finite and non-negative") != std::string_view::npos);
        REQUIRE(std::string_view(error.what()).find("expected 4 gains, got 3") != std::string_view::npos);
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
