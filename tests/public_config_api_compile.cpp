#include <concepts>
#include <expected>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sffdn/config_diagnostics.h>
#include <sffdn/fdn_config.h>

static_assert(std::same_as<decltype(sfFDN::ValidateFDNStructure(std::declval<const sfFDN::FDNConfig&>())),
                           std::expected<void, std::vector<sfFDN::ConfigIssue>>>);
static_assert(std::derived_from<sfFDN::FDNConfigError, std::runtime_error>);

void CheckPublicConfigAPI(const sfFDN::FDNConfig& config)
{
    const auto validation = sfFDN::ValidateFDNStructure(config);
    if (!validation)
    {
        for (const auto& issue : validation.error())
        {
            const auto code = issue.code;
            const auto& path = issue.path;
            const auto& message = issue.message;
            static_cast<void>(code);
            static_cast<void>(path);
            static_cast<void>(message);
        }
    }

    try
    {
        static_cast<void>(sfFDN::CreateFDNFromConfig(config));
    }
    catch (const sfFDN::FDNConfigError& error)
    {
        static_cast<void>(error.Issues());
    }
}
