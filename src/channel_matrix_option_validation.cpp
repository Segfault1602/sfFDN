#include "processor_option_validation.h"

#include "sffdn/config_diagnostics.h"
#include "sffdn/types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace
{
void AddIssue(std::vector<sfFDN::ConfigIssue>& issues, sfFDN::ConfigErrorCode code, const std::string& path,
              const std::string& message)
{
    issues.push_back({.code = code, .path = path, .message = message});
}
} // namespace

namespace sfFDN::detail
{
void ValidateOptions(const ChannelMatrixOptions& options, const std::string& path, std::vector<ConfigIssue>& issues)
{
    if (options.input_channel_count == 0U)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/input_channel_count",
                 "input channel count must be greater than zero");
    }
    if (options.output_channel_count == 0U)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/output_channel_count",
                 "output channel count must be greater than zero");
    }

    const size_t expected_count =
        static_cast<size_t>(options.input_channel_count) * static_cast<size_t>(options.output_channel_count);
    if (options.coefficients.size() != expected_count)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/coefficients",
                 "expected " + std::to_string(expected_count) + " coefficients, got " +
                     std::to_string(options.coefficients.size()));
    }
}
} // namespace sfFDN::detail
