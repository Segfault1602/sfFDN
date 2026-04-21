#pragma once

#include <nlohmann/json.hpp>

#include <sffdn/sffdn.h>

namespace sfFDN
{
nlohmann::json ToJson(const feedback_matrix_variant_t& matrix_config);
nlohmann::json ToJson(const single_channel_processor_variant_t& processor_config);
nlohmann::json ToJson(const multi_channel_processor_variant_t& processor_config);

single_channel_processor_variant_t SingleChannelProcessorFromJson(const nlohmann::json& j);
multi_channel_processor_variant_t MultichannelProcessorFromJson(const nlohmann::json& j);
feedback_matrix_variant_t FeedbackMatrixFromJson(const nlohmann::json& j);

} // namespace sfFDN