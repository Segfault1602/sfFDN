// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "processor_option_validation.h"

#include "sffdn/dattorro_delay.h"
#include "sffdn/delay_interp.h"
#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
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

std::string FieldPath(const std::string& path, const char* field)
{
    return path + "/" + field;
}

std::string IndexPath(const std::string& path, size_t index)
{
    return path + "/" + std::to_string(index);
}

bool IsKnownInterpolationType(sfFDN::DelayInterpolationType type)
{
    return type == sfFDN::DelayInterpolationType::None || type == sfFDN::DelayInterpolationType::Linear ||
           type == sfFDN::DelayInterpolationType::Allpass || type == sfFDN::DelayInterpolationType::Lagrange;
}

bool FitsUint32(double value)
{
    return value >= 0.0 && value <= static_cast<double>(std::numeric_limits<uint32_t>::max());
}

bool ValidateDelayValue(float delay, const std::string& delay_path, Issues& issues)
{
    if (delay < 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, delay_path, "delay must be non-negative");
        return false;
    }

    if (!FitsUint32(delay))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, delay_path,
                 "delay cannot be represented as a uint32_t sample count");
        return false;
    }

    return true;
}

double RequiredTap(float delay, sfFDN::DelayInterpolationType interpolation_type)
{
    double highest_tap = std::floor(static_cast<double>(delay));
    switch (interpolation_type)
    {
    case sfFDN::DelayInterpolationType::None:
        break;
    case sfFDN::DelayInterpolationType::Linear:
        highest_tap += 1.0;
        break;
    case sfFDN::DelayInterpolationType::Allpass:
    {
        const double clamped_delay = std::max(static_cast<double>(delay), 0.5);
        highest_tap = std::floor(clamped_delay);
        if (clamped_delay - highest_tap < 0.5)
        {
            highest_tap -= 1.0;
        }
        break;
    }
    case sfFDN::DelayInterpolationType::Lagrange:
        highest_tap = std::floor(std::max(static_cast<double>(delay), 1.0)) + 2.0;
        break;
    default:
        return 0.0;
    }

    return highest_tap;
}

void ValidateCapacity(double required_tap, uint32_t maximum_delay, const std::string& maximum_delay_path,
                      Issues& issues)
{
    if (!FitsUint32(required_tap))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, maximum_delay_path,
                 "interpolation taps exceed uint32_t delay capacity");
    }
    else if (required_tap > static_cast<double>(maximum_delay))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, maximum_delay_path,
                 "maximum delay is insufficient for the selected interpolation taps");
    }
}

void ValidateDelayOptions(float delay, uint32_t maximum_delay, sfFDN::DelayInterpolationType interpolation_type,
                          const std::optional<sfFDN::ModulationOptions>& lfo_config, const std::string& delay_path,
                          const std::string& maximum_delay_path, const std::string& lfo_path, bool interpolation_valid,
                          Issues& issues)
{
    if (!ValidateDelayValue(delay, delay_path, issues) || !interpolation_valid)
    {
        return;
    }

    // Delay constructs from the raw integer delay before Allpass adjusts its effective tap.
    double required_tap = std::max(std::floor(static_cast<double>(delay)), RequiredTap(delay, interpolation_type));

    if (!lfo_config.has_value())
    {
        ValidateCapacity(required_tap, maximum_delay, maximum_delay_path, issues);
        return;
    }

    sfFDN::detail::ValidateModulation(*lfo_config, lfo_path, issues);
    const float width = std::abs(lfo_config->amplitude);
    const float lower_delay = delay - width;
    const float upper_delay = delay + width;
    if (!FitsUint32(upper_delay))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, lfo_path + "/amplitude",
                 "modulation range exceeds uint32_t delay capacity");
        ValidateCapacity(required_tap, maximum_delay, maximum_delay_path, issues);
        return;
    }

    if (interpolation_type != sfFDN::DelayInterpolationType::Allpass && lower_delay < 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, lfo_path + "/amplitude",
                 "modulation range must not make the delay negative");
    }

    if (lower_delay >= 0.f)
    {
        required_tap = std::max(required_tap, RequiredTap(lower_delay, interpolation_type));
    }
    required_tap = std::max(required_tap, RequiredTap(upper_delay, interpolation_type));
    ValidateCapacity(required_tap, maximum_delay, maximum_delay_path, issues);
}
} // namespace

namespace sfFDN::detail
{
void ValidateModulation(const ModulationOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (options.frequency < 0.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, FieldPath(path, "frequency"), "frequency must be non-negative");
    }
    if (options.initial_phase < 0.f || options.initial_phase > 1.f)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, FieldPath(path, "initial_phase"), "initial phase must be in [0, 1]");
    }
}

void ValidateOptions(const DelayOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    const bool interpolation_valid = IsKnownInterpolationType(options.interp_type);
    if (!interpolation_valid)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, FieldPath(path, "interp_type"),
                 "interpolation type is unsupported");
    }
    ValidateDelayOptions(options.delay, options.max_delay, options.interp_type, options.lfo_config,
                         FieldPath(path, "delay"), FieldPath(path, "max_delay"), FieldPath(path, "lfo_config"),
                         interpolation_valid, issues);
}

void ValidateOptions(const DelayBankOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    const bool interpolation_valid = IsKnownInterpolationType(options.interpolation_type);
    if (!interpolation_valid)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, FieldPath(path, "interpolation_type"),
                 "interpolation type is unsupported");
    }

    if (options.block_size > std::numeric_limits<uint32_t>::max() / 2U)
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, FieldPath(path, "block_size"),
                 "block size cannot be doubled without overflowing uint32_t");
        return;
    }

    const uint32_t padding = options.block_size * 2U;
    for (size_t index = 0; index < options.delays.size(); ++index)
    {
        const float delay = options.delays[index];
        const std::string delay_path = IndexPath(FieldPath(path, "delays"), index);
        if (delay < 0.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, delay_path, "delay must be non-negative");
            continue;
        }

        const float maximum_delay = delay + static_cast<float>(padding);
        if (!FitsUint32(static_cast<double>(maximum_delay)))
        {
            AddIssue(issues, ConfigErrorCode::CapacityOverflow, delay_path,
                     "delay plus block padding exceeds uint32_t capacity");
            continue;
        }

        if (interpolation_valid)
        {
            ValidateDelayOptions(delay, static_cast<uint32_t>(maximum_delay), options.interpolation_type, std::nullopt,
                                 delay_path, delay_path, "", true, issues);
        }
    }
}

void ValidateOptions(const DelayBankTimeVaryingOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    const bool interpolation_valid = IsKnownInterpolationType(options.interpolation_type);
    if (!interpolation_valid)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, FieldPath(path, "interpolation_type"),
                 "interpolation type is unsupported");
    }

    if (!options.time_varying_config.empty() && options.time_varying_config.size() != options.delays.size())
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, FieldPath(path, "time_varying_config"),
                 "modulation count must match delay count");
    }

    for (size_t index = 0; index < options.delays.size(); ++index)
    {
        const std::string delay_path = IndexPath(FieldPath(path, "delays"), index);
        const std::string lfo_path = IndexPath(FieldPath(path, "time_varying_config"), index);
        const std::optional<ModulationOptions> lfo = index < options.time_varying_config.size()
                                                         ? std::optional{options.time_varying_config[index]}
                                                         : std::nullopt;
        ValidateDelayOptions(options.delays[index], options.max_delay, options.interpolation_type, lfo, delay_path,
                             FieldPath(path, "max_delay"), lfo_path, interpolation_valid, issues);
    }
}

void ValidateOptions(const DattorroDelayOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    const DelayOptions& delay_options = options.delay_config;
    const std::string delay_path = FieldPath(path, "delay_config/delay");
    if (!IsKnownInterpolationType(delay_options.interp_type))
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, FieldPath(path, "delay_config/interp_type"),
                 "interpolation type is unsupported");
    }
    if (delay_options.delay < DattorroDelay::kMinimumDelay)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, delay_path, "delay must be at least two samples");
    }

    const float width = delay_options.lfo_config.has_value() ? delay_options.lfo_config->amplitude : 0.f;
    if (delay_options.lfo_config.has_value())
    {
        ValidateModulation(*delay_options.lfo_config, FieldPath(path, "delay_config/lfo_config"), issues);
    }
    const auto delay = static_cast<double>(delay_options.delay);
    const double absolute_width = std::abs(static_cast<double>(width));
    const double maximum_tap = delay + absolute_width;
    if (delay - absolute_width < static_cast<double>(DattorroDelay::kMinimumDelay))
    {
        const std::string minimum_delay_path =
            delay_options.lfo_config.has_value() ? FieldPath(path, "delay_config/lfo_config/amplitude") : delay_path;
        AddIssue(issues, ConfigErrorCode::InvalidValue, minimum_delay_path,
                 "modulation range must keep the delay at least two samples");
    }

    if (!FitsUint32(delay) || delay > static_cast<double>(std::numeric_limits<long>::max()))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, delay_path,
                 "delay cannot be rounded safely for the feedback tap");
    }
    if (maximum_tap > static_cast<double>(std::numeric_limits<uint32_t>::max() - 64U))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, FieldPath(path, "delay_config/max_delay"),
                 "delay plus modulation width and headroom exceeds uint32_t capacity");
    }
}
} // namespace sfFDN::detail
