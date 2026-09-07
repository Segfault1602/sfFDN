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

template <typename T>
concept HasMode = requires(T value) { value.mode; };

static_assert(std::same_as<decltype(sfFDN::ValidateFDNConfig(std::declval<const sfFDN::FDNConfig&>())),
                           std::expected<void, std::vector<sfFDN::ConfigIssue>>>);
static_assert(std::derived_from<sfFDN::FDNConfigError, std::runtime_error>);
static_assert(std::same_as<decltype(sfFDN::CreateFDNFromConfig(std::declval<const sfFDN::FDNConfig&>())),
                           std::unique_ptr<sfFDN::FDN>>);
static_assert(std::same_as<decltype(sfFDN::MakeDefaultFDNConfig()), sfFDN::FDNConfig>);
static_assert(std::same_as<decltype(sfFDN::RandomizeMatrixSeeds(std::declval<sfFDN::FDNConfig&>())), void>);
static_assert(std::same_as<sfFDN::MatrixGeneratorOptions,
                           std::variant<sfFDN::ScalarMatrixType, sfFDN::VariableDiffusionOptions>>);
static_assert(std::is_aggregate_v<sfFDN::StageGainsOptions>);
static_assert(std::is_aggregate_v<sfFDN::ChannelMatrixOptions>);
static_assert(std::equality_comparable<sfFDN::ChannelMatrixOptions>);
static_assert(std::derived_from<sfFDN::ChannelMatrix, sfFDN::AudioProcessor>);
static_assert(std::is_aggregate_v<sfFDN::GeneratedMatrixOptions>);
static_assert(std::is_aggregate_v<sfFDN::InputStageConfig>);
static_assert(std::is_aggregate_v<sfFDN::OutputStageConfig>);
static_assert(std::is_aggregate_v<sfFDN::FDNConfig>);
static_assert(std::equality_comparable<sfFDN::InputStageConfig>);
static_assert(std::equality_comparable<sfFDN::OutputStageConfig>);
static_assert(std::equality_comparable<sfFDN::FDNConfig>);
static_assert(!HasMode<sfFDN::StageGainsOptions>);
static_assert(std::same_as<decltype(std::declval<const sfFDN::ScalarFeedbackMatrixOptions&>().MatrixSize()), uint32_t>);
static_assert(std::same_as<decltype(std::declval<const sfFDN::MatrixData&>().Order()), uint32_t>);
static_assert(std::equality_comparable<sfFDN::MatrixData>);
