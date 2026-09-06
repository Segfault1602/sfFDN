#include "sffdn/fdn_config.h"

#include "math_utils.h"
#include "processor_option_validation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using sfFDN::ConfigErrorCode;
using sfFDN::ConfigIssue;
using Issues = std::vector<ConfigIssue>;

void AddIssue(Issues& issues, ConfigErrorCode code, std::string path, std::string message)
{
    issues.push_back({.code = code, .path = std::move(path), .message = std::move(message)});
}

std::string IndexPath(const std::string& path, size_t index)
{
    return path + "/" + std::to_string(index);
}

void ValidatePrimaryDelayBank(const sfFDN::DelayBankOptions& options, const sfFDN::FDNConfig& config,
                              bool fdn_size_valid, bool block_size_valid, Issues& issues)
{
    constexpr const char* kPath = "/delay_bank_config";
    sfFDN::detail::ValidateOptions(options, kPath, issues);

    if (fdn_size_valid && options.delays.size() != config.fdn_size)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, std::string(kPath) + "/delays",
                 "expected " + std::to_string(config.fdn_size) + " delays, got " +
                     std::to_string(options.delays.size()));
    }

    if (!block_size_valid)
    {
        return;
    }

    if (options.block_size == 0 || options.block_size < config.block_size)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/delay_bank_config/block_size",
                 "primary delay bank block size must be at least the FDN block size");
    }

    for (size_t index = 0; index < options.delays.size(); ++index)
    {
        if (std::isfinite(options.delays[index]) && options.delays[index] >= 0.f &&
            options.delays[index] < static_cast<float>(config.block_size))
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath("/delay_bank_config/delays", index),
                     "primary delay must be at least the FDN block size");
        }
    }
}

void ValidateAttenuationFilterBank(const sfFDN::AttenuationFilterBankOptions& options, const std::string& path,
                                   uint32_t fdn_size, bool fdn_size_valid, bool allow_shared_config, Issues& issues)
{
    if (!fdn_size_valid)
    {
        return;
    }

    const size_t count = options.filter_configs.size();
    if (count != fdn_size && (!allow_shared_config || count != 1U))
    {
        const std::string expected =
            allow_shared_config ? "expected 1 or " + std::to_string(fdn_size) : "expected " + std::to_string(fdn_size);
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/AttenuationFilterBankOptions",
                 expected + " filter configurations, got " + std::to_string(count));
    }
}

void ValidateTimeVaryingMatrix(const sfFDN::TimeVaryingFeedbackMatrixOptions& options, const std::string& path,
                               uint32_t fdn_size, bool fdn_size_valid, Issues& issues)
{
    const std::string options_path = path + "/TimeVaryingFeedbackMatrixOptions";
    const bool valid_mode = options.mode == sfFDN::TimeVaryingMatrixMode::Hadamard ||
                            options.mode == sfFDN::TimeVaryingMatrixMode::RealSchur;
    if (!valid_mode)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, options_path + "/mode",
                 "time-varying matrix mode is unsupported");
    }

    const bool valid_order =
        options.matrix_size >= 2U && (options.matrix_size % 2U) == 0U &&
        (options.mode != sfFDN::TimeVaryingMatrixMode::Hadamard || sfFDN::Math::IsPowerOfTwo(options.matrix_size));
    if (!valid_order)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, options_path + "/matrix_size",
                 "matrix size must be even and at least two; Hadamard mode also requires a power of two");
    }

    if (fdn_size_valid && options.matrix_size != fdn_size)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                 "expected " + std::to_string(fdn_size) + ", got " + std::to_string(options.matrix_size));
    }

    if (options.mode == sfFDN::TimeVaryingMatrixMode::Hadamard && valid_order && !options.time_varying_config.empty() &&
        options.time_varying_config.size() != options.matrix_size / 2U)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/time_varying_config",
                 "expected " + std::to_string(options.matrix_size / 2U) + " modulation options, got " +
                     std::to_string(options.time_varying_config.size()));
    }

    for (size_t index = 0; index < options.time_varying_config.size(); ++index)
    {
        const auto& modulation = options.time_varying_config[index];
        const std::string modulation_path = IndexPath(options_path + "/time_varying_config", index);
        if (!std::isfinite(modulation.frequency))
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/frequency", "frequency must be finite");
        }
        if (!(std::abs(modulation.amplitude) <= 1.0F))
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/amplitude",
                     "amplitude must be finite and in [-1, 1]");
        }
        if (!std::isfinite(modulation.initial_phase) || modulation.initial_phase < 0.0F ||
            modulation.initial_phase > 1.0F)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/initial_phase",
                     "initial phase must be finite and in [0, 1]");
        }
    }
}

void ValidateScalarMatrix(const sfFDN::ScalarFeedbackMatrixOptions& options, const std::string& path, uint32_t fdn_size,
                          bool fdn_size_valid, Issues& issues)
{
    const std::string options_path = path + "/ScalarFeedbackMatrixOptions";
    const bool size_matches = !fdn_size_valid || options.matrix_size == fdn_size;
    if (fdn_size_valid && !size_matches)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                 "expected " + std::to_string(fdn_size) + ", got " + std::to_string(options.matrix_size));
    }

    if (size_matches && options.custom_matrix.has_value())
    {
        const uint64_t expected_count = static_cast<uint64_t>(options.matrix_size) * options.matrix_size;
        if (static_cast<uint64_t>(options.custom_matrix->size()) != expected_count)
        {
            AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/custom_matrix",
                     "expected " + std::to_string(expected_count) + " elements, got " +
                         std::to_string(options.custom_matrix->size()));
        }
    }

    if (size_matches && !options.custom_matrix.has_value() && options.type == sfFDN::ScalarMatrixType::Hadamard &&
        !sfFDN::Math::IsPowerOfTwo(options.matrix_size))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, options_path + "/matrix_size",
                 "Hadamard feedback matrices require a power-of-two size");
    }
}

void ValidateCascadedMatrix(const sfFDN::CascadedFeedbackMatrixOptions& options, const std::string& path,
                            uint32_t fdn_size, bool fdn_size_valid, Issues& issues)
{
    if (fdn_size_valid && options.matrix_size != fdn_size)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/CascadedFeedbackMatrixInfo/matrix_size",
                 "expected " + std::to_string(fdn_size) + ", got " + std::to_string(options.matrix_size));
    }
}

void ValidateFeedbackMatrix(const sfFDN::feedback_matrix_variant_t& options, const std::string& path, uint32_t fdn_size,
                            bool fdn_size_valid, Issues& issues)
{
    std::visit(sfFDN::overloaded{[&](const sfFDN::CascadedFeedbackMatrixOptions& value) {
                                     ValidateCascadedMatrix(value, path, fdn_size, fdn_size_valid, issues);
                                 },
                                 [&](const sfFDN::ScalarFeedbackMatrixOptions& value) {
                                     ValidateScalarMatrix(value, path, fdn_size, fdn_size_valid, issues);
                                 },
                                 [&](const sfFDN::TimeVaryingFeedbackMatrixOptions& value) {
                                     ValidateTimeVaryingMatrix(value, path, fdn_size, fdn_size_valid, issues);
                                 }},
               options);
}

bool IsKnownParallelGainsMode(sfFDN::ParallelGainsMode mode)
{
    return mode == sfFDN::ParallelGainsMode::Split || mode == sfFDN::ParallelGainsMode::Merge ||
           mode == sfFDN::ParallelGainsMode::Parallel;
}

void ValidateSingleChannelProcessor(const sfFDN::single_channel_processor_variant_t& options, const std::string& path,
                                    Issues& issues)
{
    std::visit(sfFDN::overloaded{
                   [&](const sfFDN::DelayOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/DelayOptions", issues);
                   },
                   [&](const sfFDN::DattorroDelayOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/DattorroDelayOptions", issues);
                   },
                   [&](const sfFDN::SchroederAllpassSectionOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/SchroederAllpassSectionOptions", issues);
                   },
                   [&](const sfFDN::TimeVaryingSchroederAllpassSectionOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/TimeVaryingSchroederAllpassSectionOptions",
                                                      issues);
                   },
                   [](const sfFDN::AllpassFilterOptions&) {},
                   [](const sfFDN::CascadedBiquadsOptions&) {},
                   [](const sfFDN::FirOptions&) {},
                   [](const sfFDN::GraphicEQOptions&) {},
                   [](const sfFDN::ControllableFullWaveRectifierOptions&) {},
                   [](const sfFDN::SignalDependentFractionalDelayOptions&) {},
                   [](const sfFDN::RingModulatorOptions&) {},
               },
               options);
}

void ValidateMultichannelProcessor(const sfFDN::multi_channel_processor_variant_t& options, const std::string& path,
                                   uint32_t fdn_size, bool fdn_size_valid, bool allow_shared_attenuation,
                                   Issues& issues)
{
    std::visit(sfFDN::overloaded{
                   [&](const sfFDN::ParallelGainsOptions& value) {
                       const std::string options_path = path + "/ParallelGainsConfig";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (IsKnownParallelGainsMode(value.mode) && value.mode != sfFDN::ParallelGainsMode::Parallel)
                       {
                           AddIssue(issues, ConfigErrorCode::UnsupportedValue, options_path + "/mode",
                                    "parallel gains in a multichannel processor block must use Parallel mode");
                       }
                       if (fdn_size_valid && value.gains.size() != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/gains",
                                    "expected " + std::to_string(fdn_size) + " gains, got " +
                                        std::to_string(value.gains.size()));
                       }
                   },
                   [&](const sfFDN::MultichannelProcessorOptions& value) {
                       const std::string options_path = path + "/MultichannelProcessorOptions";
                       if (fdn_size_valid && value.channels.size() != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/channels",
                                    "expected " + std::to_string(fdn_size) + " channels, got " +
                                        std::to_string(value.channels.size()));
                       }
                       for (size_t index = 0; index < value.channels.size(); ++index)
                       {
                           if (value.channels[index].has_value())
                           {
                               ValidateSingleChannelProcessor(*value.channels[index],
                                                              IndexPath(options_path + "/channels", index), issues);
                           }
                       }
                   },
                   [&](const sfFDN::AttenuationFilterBankOptions& value) {
                       ValidateAttenuationFilterBank(value, path, fdn_size, fdn_size_valid, allow_shared_attenuation,
                                                     issues);
                   },
                   [&](const sfFDN::DelayBankOptions& value) {
                       const std::string options_path = path + "/DelayBankOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.delays.size() != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/delays",
                                    "expected " + std::to_string(fdn_size) + " delays, got " +
                                        std::to_string(value.delays.size()));
                       }
                   },
                   [&](const sfFDN::DelayBankTimeVaryingOptions& value) {
                       const std::string options_path = path + "/DelayBankTimeVaryingOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.delays.size() != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/delays",
                                    "expected " + std::to_string(fdn_size) + " delays, got " +
                                        std::to_string(value.delays.size()));
                       }
                   },
                   [&](const sfFDN::CascadedFeedbackMatrixOptions& value) {
                       ValidateCascadedMatrix(value, path, fdn_size, fdn_size_valid, issues);
                   },
                   [&](const sfFDN::ScalarFeedbackMatrixOptions& value) {
                       ValidateScalarMatrix(value, path, fdn_size, fdn_size_valid, issues);
                   },
               },
               options);
}

void ValidateGains(const sfFDN::ParallelGainsOptions& options, const std::string& path,
                   sfFDN::ParallelGainsMode expected_mode, uint32_t fdn_size, bool fdn_size_valid, Issues& issues)
{
    sfFDN::detail::ValidateOptions(options, path, issues);
    if (IsKnownParallelGainsMode(options.mode) && options.mode != expected_mode)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, path + "/mode",
                 std::string("expected ") + (expected_mode == sfFDN::ParallelGainsMode::Split ? "Split" : "Merge") +
                     " mode");
    }
    if (fdn_size_valid && options.gains.size() != fdn_size)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/gains",
                 "expected " + std::to_string(fdn_size) + " gains, got " + std::to_string(options.gains.size()));
    }
}

} // namespace

namespace sfFDN
{

FDNConfigError::FDNConfigError(std::vector<ConfigIssue> issues)
    : std::runtime_error(BuildMessage(issues))
    , issues_(std::move(issues))
{
}

const std::vector<ConfigIssue>& FDNConfigError::Issues() const noexcept
{
    return issues_;
}

std::string FDNConfigError::BuildMessage(const std::vector<ConfigIssue>& issues)
{
    std::string message = "Invalid FDNConfig";
    for (const auto& issue : issues)
    {
        message += "\n";
        message += issue.path;
        message += ": ";
        message += issue.message;
    }
    return message;
}

std::expected<void, std::vector<ConfigIssue>> ValidateFDNStructure(const FDNConfig& config)
{
    Issues issues;
    const bool fdn_size_valid = config.fdn_size > 0;
    const bool block_size_valid = config.block_size > 0;
    if (!fdn_size_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/fdn_size", "FDN size must be greater than zero");
    }
    if (!block_size_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/block_size", "block size must be greater than zero");
    }
    if (!std::isfinite(config.sample_rate) || config.sample_rate <= 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/sample_rate", "sample rate must be finite and positive");
    }
    if (!std::isfinite(config.direct_gain))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/direct_gain", "direct gain must be finite");
    }
    if (fdn_size_valid && block_size_valid &&
        config.fdn_size > std::numeric_limits<uint32_t>::max() / config.block_size)
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, "/fdn_size",
                 "FDN size multiplied by block size exceeds uint32_t indexing capacity");
    }

    ValidatePrimaryDelayBank(config.delay_bank_config, config, fdn_size_valid, block_size_valid, issues);
    ValidateGains(config.input_block_config.parallel_gains_config, "/input_block_config/parallel_gains_config",
                  ParallelGainsMode::Split, config.fdn_size, fdn_size_valid, issues);
    ValidateGains(config.output_block_config.parallel_gains_config, "/output_block_config/parallel_gains_config",
                  ParallelGainsMode::Merge, config.fdn_size, fdn_size_valid, issues);
    ValidateFeedbackMatrix(config.feedback_matrix_config, "/feedback_matrix_config", config.fdn_size, fdn_size_valid,
                           issues);

    for (size_t index = 0; index < config.input_block_config.single_channel_processors.size(); ++index)
    {
        ValidateSingleChannelProcessor(config.input_block_config.single_channel_processors[index],
                                       IndexPath("/input_block_config/single_channel_processors", index), issues);
    }

    for (size_t index = 0; index < config.input_block_config.multichannel_processors.size(); ++index)
    {
        ValidateMultichannelProcessor(config.input_block_config.multichannel_processors[index],
                                      IndexPath("/input_block_config/multichannel_processors", index), config.fdn_size,
                                      fdn_size_valid, false, issues);
    }

    for (size_t index = 0; index < config.output_block_config.multichannel_processors.size(); ++index)
    {
        ValidateMultichannelProcessor(config.output_block_config.multichannel_processors[index],
                                      IndexPath("/output_block_config/multichannel_processors", index), config.fdn_size,
                                      fdn_size_valid, false, issues);
    }

    for (size_t index = 0; index < config.output_block_config.single_channel_processors.size(); ++index)
    {
        ValidateSingleChannelProcessor(config.output_block_config.single_channel_processors[index],
                                       IndexPath("/output_block_config/single_channel_processors", index), issues);
    }

    if (config.attenuation_filter_bank_config.has_value())
    {
        ValidateAttenuationFilterBank(*config.attenuation_filter_bank_config, "/attenuation_filter_bank_config",
                                      config.fdn_size, fdn_size_valid, true, issues);
    }

    for (size_t index = 0; index < config.tone_correction_filters.size(); ++index)
    {
        ValidateSingleChannelProcessor(config.tone_correction_filters[index],
                                       IndexPath("/tone_correction_filters", index), issues);
    }

    for (size_t index = 0; index < config.loop_filter_configs.size(); ++index)
    {
        ValidateMultichannelProcessor(config.loop_filter_configs[index], IndexPath("/loop_filter_configs", index),
                                      config.fdn_size, fdn_size_valid, true, issues);
    }

    if (!issues.empty())
    {
        return std::unexpected(std::move(issues));
    }
    return {};
}

} // namespace sfFDN
