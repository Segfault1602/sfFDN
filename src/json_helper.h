#pragma once

#include <nlohmann/json.hpp>

#include <sffdn/sffdn.h>

namespace sfFDN
{
void ThrowIfNotType(const nlohmann::json& j, const std::string& expected_type);
void ThrowIfDoesNotContainKey(const nlohmann::json& j, const std::string& key);

std::unique_ptr<AudioProcessor> from_json(const nlohmann::json& j);

nlohmann::json ToJson(const feedback_matrix_variant_t& matrix_config);
nlohmann::json ToJson(const single_channel_processor_variant_t& processor_config);
nlohmann::json ToJson(const multichannel_processor_variant_t& processor_config);

single_channel_processor_variant_t SingleChannelProcessorFromJson(const nlohmann::json& j);
multichannel_processor_variant_t MultichannelProcessorFromJson(const nlohmann::json& j);
feedback_matrix_variant_t FeedbackMatrixFromJson(const nlohmann::json& j);

} // namespace sfFDN