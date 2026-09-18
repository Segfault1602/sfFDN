#pragma once

#include "sffdn/config_diagnostics.h"
#include "sffdn/types.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace sfFDN::detail
{

void ValidateOptions(const DelayOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const DelayBankOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const DelayBankTimeVaryingOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const DattorroDelayOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const ParallelGainsOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const StageGainsOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const ChannelMatrixOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const SchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TimeVaryingSchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const ScalarFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const GeneratedMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const VariableDiffusionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const CascadedFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TimeVaryingFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const FilterCoefficients& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const AllpassFilterOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const CascadedBiquadsOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const FirOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const GraphicEQOptions& options, float sample_rate, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const HomogenousFilterOptions& options, float sample_rate, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TwoBandFilterOptions& options, float sample_rate, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const ThreeBandFilterOptions& options, float sample_rate, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TenBandFilterOptions& options, float sample_rate, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const ControllableFullWaveRectifierOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const SignalDependentFractionalDelayOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const RingModulatorOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateAttenuationOptions(const attenuation_filter_variant_t& options, float sample_rate, const std::string& path,
                                std::vector<ConfigIssue>& issues, bool allow_inferred_delay);
void ValidateModulation(const ModulationOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);

template <class Options>
const Options& RequireValidOptions(const Options& options)
{
    std::vector<ConfigIssue> issues;
    ValidateOptions(options, "", issues);
    if (!issues.empty())
    {
        throw std::invalid_argument(issues.front().path + ": " + issues.front().message);
    }
    // The deleted rvalue overload below prevents references to temporaries.
    return options; // NOLINT(bugprone-return-const-ref-from-parameter)
}

template <class Options>
const Options& RequireValidOptions(const Options&& options) = delete;

template <class Options>
const Options& RequireValidOptions(const Options& options, float sample_rate)
{
    std::vector<ConfigIssue> issues;
    ValidateOptions(options, sample_rate, "", issues);
    if (!issues.empty())
    {
        throw std::invalid_argument(issues.front().path + ": " + issues.front().message);
    }
    return options; // NOLINT(bugprone-return-const-ref-from-parameter)
}

template <class Options>
const Options& RequireValidOptions(const Options&& options, float sample_rate) = delete;

} // namespace sfFDN::detail
