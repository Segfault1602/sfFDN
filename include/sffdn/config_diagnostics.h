#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfFDN
{

enum class ConfigErrorCode : uint8_t
{
    InvalidValue,
    SizeMismatch,
    UnsupportedValue,
    CapacityOverflow
};

struct ConfigIssue
{
    ConfigErrorCode code{ConfigErrorCode::InvalidValue};
    std::string path;
    std::string message;

    bool operator==(const ConfigIssue&) const = default;
};

class FDNConfigError : public std::runtime_error
{
  public:
    explicit FDNConfigError(std::vector<ConfigIssue> issues);

    [[nodiscard]] const std::vector<ConfigIssue>& Issues() const noexcept;

  private:
    static std::string BuildMessage(const std::vector<ConfigIssue>& issues);

    std::vector<ConfigIssue> issues_;
};

} // namespace sfFDN
