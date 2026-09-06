#include "processor_option_validation.h"

#include "sffdn/config_diagnostics.h"
#include "sffdn/types.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
void AddIssue(std::vector<sfFDN::ConfigIssue>& issues, sfFDN::ConfigErrorCode code, const std::string& path,
              const std::string& message)
{
    issues.push_back({.code = code, .path = path, .message = message});
}

std::string IndexPath(const std::string& path, size_t index)
{
    return path + "/" + std::to_string(index);
}

bool IsSupportedMode(sfFDN::ParallelGainsMode mode)
{
    return mode == sfFDN::ParallelGainsMode::Split || mode == sfFDN::ParallelGainsMode::Merge ||
           mode == sfFDN::ParallelGainsMode::Parallel;
}
} // namespace

namespace sfFDN::detail
{
void ValidateOptions(const ParallelGainsOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (!IsSupportedMode(options.mode))
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, path + "/mode", "parallel gains mode is unsupported");
    }

    if (!options.time_varying_config.empty() && options.time_varying_config.size() != options.gains.size())
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/time_varying_config",
                 "expected " + std::to_string(options.gains.size()) + " modulation options, got " +
                     std::to_string(options.time_varying_config.size()));
    }

    for (size_t index = 0; index < options.time_varying_config.size(); ++index)
    {
        const auto& modulation = options.time_varying_config[index];
        const std::string modulation_path = IndexPath(path + "/time_varying_config", index);
        ValidateModulation(modulation, modulation_path, issues);

        if (index >= options.gains.size())
        {
            continue;
        }

        const double center = options.gains[index];
        const double amplitude = std::abs(static_cast<double>(modulation.amplitude));
        constexpr double kMaximumGain = std::numeric_limits<float>::max();
        if (center + amplitude > kMaximumGain || center - amplitude < -kMaximumGain)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path + "/gains", index),
                     "center gain plus or minus modulation amplitude exceeds float range");
        }
    }
}

void ValidateOptions(const SchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.delays.empty())
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/delays", "at least one delay is required");
    }

    if (options.gains.size() != options.delays.size())
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/gains",
                 "expected " + std::to_string(options.delays.size()) + " gains, got " +
                     std::to_string(options.gains.size()));
    }

    constexpr uint32_t kCapacityPadding = 16U;
    constexpr uint32_t kMaximumDelay = std::numeric_limits<uint32_t>::max() - kCapacityPadding;
    for (size_t index = 0; index < options.delays.size(); ++index)
    {
        const float delay = options.delays[index];
        if (delay < 0.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path + "/delays", index),
                     "delay must be non-negative");
        }
        else if (static_cast<double>(delay) > static_cast<double>(kMaximumDelay))
        {
            AddIssue(issues, ConfigErrorCode::CapacityOverflow, IndexPath(path + "/delays", index),
                     "delay plus capacity padding exceeds uint32_t capacity");
        }
    }
}

void ValidateOptions(const TimeVaryingSchroederAllpassSectionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.delays.empty())
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/delays", "at least one delay is required");
    }

    if (options.gains.size() != options.delays.size())
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/gains",
                 "expected " + std::to_string(options.delays.size()) + " gains, got " +
                     std::to_string(options.gains.size()));
    }

    if (options.time_varying_config.size() != options.delays.size())
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/time_varying_config",
                 "expected " + std::to_string(options.delays.size()) + " modulation options, got " +
                     std::to_string(options.time_varying_config.size()));
    }

    constexpr uint32_t kMaximumDelay = std::numeric_limits<uint32_t>::max();
    for (size_t index = 0; index < options.delays.size(); ++index)
    {
        const float delay = options.delays[index];
        if (delay <= 0.f || std::trunc(delay) != delay)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path + "/delays", index),
                     "delay must be a positive integer");
        }
        else if (static_cast<double>(delay) > static_cast<double>(kMaximumDelay))
        {
            AddIssue(issues, ConfigErrorCode::CapacityOverflow, IndexPath(path + "/delays", index),
                     "delay exceeds uint32_t capacity");
        }
    }

    for (size_t index = 0; index < options.time_varying_config.size(); ++index)
    {
        const auto& modulation = options.time_varying_config[index];
        const std::string modulation_path = IndexPath(path + "/time_varying_config", index);
        ValidateModulation(modulation, modulation_path, issues);

        if (modulation.frequency == 0.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/frequency",
                     "frequency must be strictly positive");
        }
        if (modulation.amplitude == 0.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/amplitude",
                     "amplitude must be non-zero");
        }
    }

    for (size_t index = 0; index < options.gains.size(); ++index)
    {
        if (index >= options.time_varying_config.size())
        {
            continue;
        }

        const float gain = options.gains[index];
        const float amplitude = options.time_varying_config[index].amplitude;
        if (std::abs(gain) + std::abs(amplitude) >= 1.f)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, IndexPath(path + "/gains", index),
                     "absolute gain plus modulation amplitude must be less than one");
        }
    }
}
} // namespace sfFDN::detail
