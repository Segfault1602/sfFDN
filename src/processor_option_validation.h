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
void ValidateOptions(const SchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TimeVaryingSchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const ScalarFeedbackMatrixOptions& options, const std::string& path,
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
void ValidateOptions(const GraphicEQOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const HomogenousFilterOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TwoBandFilterOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const ThreeBandFilterOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const TenBandFilterOptions& options, const std::string& path, std::vector<ConfigIssue>& issues);
void ValidateOptions(const ControllableFullWaveRectifierOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const SignalDependentFractionalDelayOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateOptions(const RingModulatorOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues);
void ValidateAttenuationOptions(const attenuation_filter_variant_t& options, const std::string& path,
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
    return options;
}

} // namespace sfFDN::detail
