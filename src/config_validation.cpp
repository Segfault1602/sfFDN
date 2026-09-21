#include "sffdn/fdn_config.h"

#include "processor_option_validation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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

void ValidateScalarMatrixDimension(const sfFDN::ScalarFeedbackMatrixOptions& options, const std::string& path,
                                   uint32_t fdn_size, bool fdn_size_valid, Issues& issues)
{
    std::visit(sfFDN::overloaded{
                   [&](const sfFDN::GeneratedMatrixOptions& source) {
                       if (fdn_size_valid && source.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch,
                                    path + "/source/GeneratedMatrixOptions/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(source.matrix_size));
                       }
                   },
                   [&](const sfFDN::MatrixData& source) {
                       if (fdn_size_valid && source.Order() != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/source/MatrixData/order",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(source.Order()));
                       }
                   },
               },
               options.source);
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
    const size_t count = options.filter_configs.size();
    const std::string options_path = path + "/AttenuationFilterBankOptions";
    if (fdn_size_valid && count != fdn_size && (!allow_shared_config || count != 1U))
    {
        const std::string expected =
            allow_shared_config ? "expected 1 or " + std::to_string(fdn_size) : "expected " + std::to_string(fdn_size);
        AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path,
                 expected + " filter configurations, got " + std::to_string(count));
    }

    for (size_t index = 0; index < count; ++index)
    {
        sfFDN::detail::ValidateAttenuationOptions(options.filter_configs[index], IndexPath(options_path, index), issues,
                                                  allow_shared_config);
    }
}

void ValidateFeedbackMatrix(const sfFDN::feedback_matrix_variant_t& options, const std::string& path, uint32_t fdn_size,
                            bool fdn_size_valid, Issues& issues)
{
    std::visit(sfFDN::overloaded{
                   [&](const sfFDN::CascadedFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/CascadedFeedbackMatrixInfo";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix_size));
                       }
                   },
                   [&](const sfFDN::ScalarFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/ScalarFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       ValidateScalarMatrixDimension(value, options_path, fdn_size, fdn_size_valid, issues);
                   },
                   [&](const sfFDN::TimeVaryingFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/TimeVaryingFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix_size));
                       }
                   },
                   [&](const sfFDN::KroneckerFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/KroneckerFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix_size));
                       }
                   },
                   [&](const sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/TimeVaryingKroneckerFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix.matrix_size));
                       }
                   },
               },
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
                   [&](const sfFDN::AllpassFilterOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/AllpassFilterOptions", issues);
                   },
                   [&](const sfFDN::CascadedBiquadsOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/CascadedBiquadsOptions", issues);
                   },
                   [&](const sfFDN::FirOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/FirOptions", issues);
                   },
                   [&](const sfFDN::GraphicEQOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/GraphicEQOptions", issues);
                   },
                   [&](const sfFDN::ControllableFullWaveRectifierOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/ControllableFullWaveRectifierOptions", issues);
                   },
                   [&](const sfFDN::SignalDependentFractionalDelayOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/SignalDependentFractionalDelayOptions", issues);
                   },
                   [&](const sfFDN::RingModulatorOptions& value) {
                       sfFDN::detail::ValidateOptions(value, path + "/RingModulatorOptions", issues);
                   },
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
                       const std::string options_path = path + "/CascadedFeedbackMatrixInfo";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix_size));
                       }
                   },
                   [&](const sfFDN::ScalarFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/ScalarFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       ValidateScalarMatrixDimension(value, options_path, fdn_size, fdn_size_valid, issues);
                   },
                   [&](const sfFDN::KroneckerFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/KroneckerFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix_size));
                       }
                   },
                   [&](const sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions& value) {
                       const std::string options_path = path + "/TimeVaryingKroneckerFeedbackMatrixOptions";
                       sfFDN::detail::ValidateOptions(value, options_path, issues);
                       if (fdn_size_valid && value.matrix.matrix_size != fdn_size)
                       {
                           AddIssue(issues, ConfigErrorCode::SizeMismatch, options_path + "/matrix/matrix_size",
                                    "expected " + std::to_string(fdn_size) + ", got " +
                                        std::to_string(value.matrix.matrix_size));
                       }
                   },
               },
               options);
}

void ValidateGains(const sfFDN::StageGainsOptions& options, const std::string& path, uint32_t fdn_size,
                   bool fdn_size_valid, Issues& issues)
{
    sfFDN::detail::ValidateOptions(options, path, issues);
    if (fdn_size_valid && options.gains.size() != fdn_size)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/gains",
                 "expected " + std::to_string(fdn_size) + " gains, got " + std::to_string(options.gains.size()));
    }
}

void ValidateChannelMatrix(const sfFDN::ChannelMatrixOptions& options, const std::string& path, uint32_t expected_input,
                           bool expected_input_valid, uint32_t expected_output, bool expected_output_valid,
                           Issues& issues)
{
    sfFDN::detail::ValidateOptions(options, path, issues);
    if (expected_input_valid && options.input_channel_count != expected_input)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/input_channel_count",
                 "expected " + std::to_string(expected_input) + ", got " + std::to_string(options.input_channel_count));
    }
    if (expected_output_valid && options.output_channel_count != expected_output)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/output_channel_count",
                 "expected " + std::to_string(expected_output) + ", got " +
                     std::to_string(options.output_channel_count));
    }
}

/** Validates one boundary stage: either an explicit matrix or the stage gains, never both. */
void ValidateBoundaryStage(const std::optional<sfFDN::ChannelMatrixOptions>& matrix,
                           const sfFDN::StageGainsOptions& gains, const std::string& stage_path,
                           uint32_t external_count, bool external_count_valid, uint32_t fdn_size, bool fdn_size_valid,
                           bool is_input, Issues& issues)
{
    const std::string matrix_path = stage_path + "/boundary_matrix";
    const std::string gains_path = stage_path + "/parallel_gains_config";

    if (matrix.has_value())
    {
        const uint32_t expected_input = is_input ? external_count : fdn_size;
        const uint32_t expected_output = is_input ? fdn_size : external_count;
        const bool expected_input_valid = is_input ? external_count_valid : fdn_size_valid;
        const bool expected_output_valid = is_input ? fdn_size_valid : external_count_valid;
        ValidateChannelMatrix(*matrix, matrix_path, expected_input, expected_input_valid, expected_output,
                              expected_output_valid, issues);

        if (!gains.gains.empty() || !gains.time_varying_config.empty())
        {
            AddIssue(issues, ConfigErrorCode::UnsupportedValue, gains_path,
                     "stage gains must be empty when a boundary matrix is set");
        }
        return;
    }

    if (external_count_valid && external_count != 1U)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, matrix_path,
                 std::string(is_input ? "an input" : "an output") + " channel count of " +
                     std::to_string(external_count) + " requires a boundary matrix");
    }
    ValidateGains(gains, gains_path, fdn_size, fdn_size_valid, issues);
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
        message += '\n';
        message += issue.path;
        message += ": ";
        message += issue.message;
    }
    return message;
}

std::expected<void, std::vector<ConfigIssue>> ValidateFDNConfig(const FDNConfig& config)
{
    Issues issues;
    const bool fdn_size_valid = config.fdn_size > 0;
    const bool block_size_valid = config.block_size > 0;
    const bool input_channels_valid = config.input_channel_count > 0;
    const bool output_channels_valid = config.output_channel_count > 0;
    if (!fdn_size_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/fdn_size", "FDN size must be greater than zero");
    }
    if (!input_channels_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/input_channel_count",
                 "input channel count must be greater than zero");
    }
    if (!output_channels_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, "/output_channel_count",
                 "output channel count must be greater than zero");
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
    ValidateBoundaryStage(config.input_block_config.boundary_matrix, config.input_block_config.parallel_gains_config,
                          "/input_block_config", config.input_channel_count, input_channels_valid, config.fdn_size,
                          fdn_size_valid, true, issues);
    ValidateBoundaryStage(config.output_block_config.boundary_matrix, config.output_block_config.parallel_gains_config,
                          "/output_block_config", config.output_channel_count, output_channels_valid, config.fdn_size,
                          fdn_size_valid, false, issues);

    if (config.direct_matrix.has_value())
    {
        ValidateChannelMatrix(*config.direct_matrix, "/direct_matrix", config.input_channel_count,
                              input_channels_valid, config.output_channel_count, output_channels_valid, issues);
        if (config.direct_gain != 0.f)
        {
            AddIssue(issues, ConfigErrorCode::UnsupportedValue, "/direct_gain",
                     "direct gain must be zero when a direct matrix is set");
        }
    }
    else if (input_channels_valid && output_channels_valid &&
             config.input_channel_count != config.output_channel_count && config.direct_gain != 0.f)
    {
        // The scalar gain is a diagonal path, so it cannot bridge differing input and output channel counts.
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, "/direct_gain",
                 "a nonzero scalar direct gain requires matching input and output channel counts; use a direct matrix "
                 "instead");
    }

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
