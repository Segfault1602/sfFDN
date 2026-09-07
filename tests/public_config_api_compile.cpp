#include <concepts>
#include <expected>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sffdn/config_diagnostics.h>
#include <sffdn/fdn_config.h>
#include <sffdn/sffdn.h>

#ifdef NLOHMANN_JSON_VERSION_MAJOR
#error "Core public headers must not include nlohmann/json"
#endif

static_assert(std::same_as<decltype(sfFDN::ValidateFDNConfig(std::declval<const sfFDN::FDNConfig&>())),
                           std::expected<void, std::vector<sfFDN::ConfigIssue>>>);
static_assert(std::derived_from<sfFDN::FDNConfigError, std::runtime_error>);
static_assert(std::same_as<decltype(sfFDN::CreateFDNFromConfig(std::declval<const sfFDN::FDNConfig&>())),
                           std::unique_ptr<sfFDN::FDN>>);
