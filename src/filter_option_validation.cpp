// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "processor_option_validation.h"

#include "sffdn/config_diagnostics.h"
#include "sffdn/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace
{
using sfFDN::ConfigErrorCode;
using sfFDN::ConfigIssue;
using Issues = std::vector<ConfigIssue>;

void AddIssue(Issues& issues, ConfigErrorCode code, const std::string& path, const char* message)
{
    issues.push_back({.code = code, .path = path, .message = message});
}

std::string IndexPath(const std::string& path, size_t index)
{
    return path + "/" + std::to_string(index);
}

bool IsFloatRepresentable(double value)
{
    return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
}

void ValidateDelay(float delay, const std::string& path, bool allow_inferred_delay, Issues& issues)
{
    if (allow_inferred_delay && delay <= 0.f)
    {
        return;
    }
    if (delay < 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path, "delay must be non-negative");
    }
}

void ValidateT60s(std::span<const float> t60s, const std::string& path, Issues& issues)
{
    for (size_t index = 0; index < t60s.size(); ++index)
    {
        if (t60s[index] <= 0.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path, index), "T60 must be positive");
        }
    }
}

void ValidateSampleRate(float sample_rate, const std::string& path, Issues& issues)
{
    if (sample_rate <= 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path, "sample rate must be positive");
    }
}

void ValidateAttenuation(const sfFDN::HomogenousFilterOptions& options, const std::string& path,
                         bool allow_inferred_delay, Issues& issues)
{
    if (options.t60 <= 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/t60", "T60 must be positive");
    }
    ValidateDelay(options.delay, path + "/delay", allow_inferred_delay, issues);
    ValidateSampleRate(options.sample_rate, path + "/sample_rate", issues);
}

void ValidateAttenuation(const sfFDN::TwoBandFilterOptions& options, const std::string& path,
                         bool allow_inferred_delay, Issues& issues)
{
    ValidateT60s(options.t60s, path + "/t60s", issues);
    ValidateDelay(options.delay, path + "/delay", allow_inferred_delay, issues);
    ValidateSampleRate(options.sample_rate, path + "/sample_rate", issues);
}

void ValidateAttenuation(const sfFDN::ThreeBandFilterOptions& options, const std::string& path,
                         bool allow_inferred_delay, Issues& issues)
{
    ValidateT60s(options.t60s, path + "/t60s", issues);
    ValidateDelay(options.delay, path + "/delay", allow_inferred_delay, issues);
    ValidateSampleRate(options.sample_rate, path + "/sample_rate", issues);

    const float nyquist = options.sample_rate * 0.5f;
    if (options.freqs[0] <= 0.f || options.freqs[0] >= options.freqs[1] || options.freqs[1] >= nyquist)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/freqs",
                 "frequencies must be strictly increasing and below Nyquist");
    }
    if (options.q <= 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/q", "Q must be positive");
    }
}

void ValidateAttenuation(const sfFDN::TenBandFilterOptions& options, const std::string& path,
                         bool allow_inferred_delay, Issues& issues)
{
    ValidateT60s(options.t60s, path + "/t60s", issues);
    ValidateDelay(options.delay, path + "/delay", allow_inferred_delay, issues);
    ValidateSampleRate(options.sample_rate, path + "/sample_rate", issues);

    if (options.sample_rate <= 32000.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/sample_rate",
                 "sample rate must exceed 32000 Hz for the 16000 Hz band");
    }

    const float nyquist = options.sample_rate * 0.5f;
    if (options.shelf_cutoff <= 0.f || options.shelf_cutoff >= nyquist)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/shelf_cutoff",
                 "shelf cutoff must be strictly between zero and Nyquist");
    }
}
} // namespace

namespace sfFDN::detail
{
void ValidateOptions(const FilterCoefficients& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (options.a0 == 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/a0", "a0 must be non-zero for normalization");
        return;
    }

    constexpr std::array<std::pair<const char*, float FilterCoefficients::*>, 5> kNormalizedMembers = {
        {
            {"b0", &FilterCoefficients::b0},
            {"b1", &FilterCoefficients::b1},
            {"b2", &FilterCoefficients::b2},
            {"a1", &FilterCoefficients::a1},
            {"a2", &FilterCoefficients::a2},
        },
    };
    for (const auto& [name, member] : kNormalizedMembers)
    {
        if (!IsFloatRepresentable(static_cast<double>(options.*member) / static_cast<double>(options.a0)))
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/" + name,
                     "normalized coefficient cannot be represented as a float");
        }
    }
}

void ValidateOptions(const AllpassFilterOptions& /*options*/, const std::string& /*path*/,
                     std::vector<ConfigIssue>& /*issues*/)
{
    // Coefficients outside the conventional stable range remain available for experimental use.
}

void ValidateOptions(const CascadedBiquadsOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.coeffs.size() > std::numeric_limits<uint32_t>::max())
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, path + "/coeffs",
                 "stage count cannot be represented as uint32_t");
    }

    for (size_t index = 0; index < options.coeffs.size(); ++index)
    {
        ValidateOptions(options.coeffs[index], IndexPath(path + "/coeffs", index), issues);
    }
}

void ValidateOptions(const FirOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (options.coeffs.empty())
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/coeffs", "at least one FIR coefficient is required");
        return;
    }

    constexpr size_t kMaximumTapCount =
        std::min(static_cast<size_t>(std::numeric_limits<int>::max()),
                 static_cast<size_t>(std::numeric_limits<uint32_t>::max() / 2U));
    if (options.coeffs.size() > kMaximumTapCount)
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, path + "/coeffs",
                 "tap count exceeds FIR storage or backend indexing capacity");
    }
}

void ValidateOptions(const GraphicEQOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    ValidateSampleRate(options.sample_rate, path + "/sample_rate", issues);
    const float nyquist = options.sample_rate * 0.5f;
    if (8000.f >= nyquist)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/sample_rate",
                 "sample rate must place the 8000 Hz shelf below Nyquist");
    }

    for (size_t index = 0; index < options.freqs.size(); ++index)
    {
        const bool increasing = index == 0 || options.freqs[index - 1] < options.freqs[index];
        if (options.freqs[index] <= 0.f || !increasing || options.freqs[index] >= nyquist)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path + "/freqs", index),
                     "frequency must be strictly increasing and below Nyquist");
        }
    }
}

void ValidateOptions(const HomogenousFilterOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    ValidateAttenuation(options, path, false, issues);
}

void ValidateOptions(const TwoBandFilterOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    ValidateAttenuation(options, path, false, issues);
}

void ValidateOptions(const ThreeBandFilterOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    ValidateAttenuation(options, path, false, issues);
}

void ValidateOptions(const TenBandFilterOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    ValidateAttenuation(options, path, false, issues);
}

void ValidateOptions(const ControllableFullWaveRectifierOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.alpha < 0.f || options.alpha > 1.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/alpha", "alpha must be in [0, 1]");
    }
    if (options.dc_block && options.sample_rate <= 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/sample_rate",
                 "sample rate must be positive when dc_block is enabled");
    }
}

void ValidateOptions(const SignalDependentFractionalDelayOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.d < 0.f || options.d > 1.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/d", "d must be in [0, 1]");
    }
}

void ValidateOptions(const RingModulatorOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (options.frequency < 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/frequency", "frequency must be non-negative");
    }
    if (options.initial_phase < 0.f || options.initial_phase > 1.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/initial_phase", "initial phase must be in [0, 1]");
    }
}

void ValidateAttenuationOptions(const attenuation_filter_variant_t& options, const std::string& path,
                                std::vector<ConfigIssue>& issues, bool allow_inferred_delay)
{
    std::visit(overloaded{
                   [&](const HomogenousFilterOptions& value) {
                       ValidateAttenuation(value, path + "/ProportionalAttenuationConfig", allow_inferred_delay,
                                           issues);
                   },
                   [&](const TwoBandFilterOptions& value) {
                       ValidateAttenuation(value, path + "/TwoBandFilterConfig", allow_inferred_delay, issues);
                   },
                   [&](const ThreeBandFilterOptions& value) {
                       ValidateAttenuation(value, path + "/ThreeBandFilterConfig", allow_inferred_delay, issues);
                   },
                   [&](const TenBandFilterOptions& value) {
                       ValidateAttenuation(value, path + "/TenBandFilterConfig", allow_inferred_delay, issues);
                   },
               },
               options);
}
} // namespace sfFDN::detail
