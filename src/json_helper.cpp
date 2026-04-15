#include "json_helper.h"

#include "sffdn/types.h"

#include <cstdint>
#include <map>

namespace sfFDN
{

void ThrowIfNotType(const nlohmann::json& j, const std::string& expected_type)
{
    ThrowIfDoesNotContainKey(j, "type");

    if (!j["type"].is_string())
    {
        throw std::invalid_argument("JSON 'type' field must be a string.");
    }

    if (j["type"] != expected_type)
    {
        std::string message =
            std::format("JSON object is of type '{}', expected '{}'.", j["type"].get<std::string>(), expected_type);
        throw std::invalid_argument(message);
    }
}

void ThrowIfDoesNotContainKey(const nlohmann::json& j, const std::string& key)
{
    if (!j.contains(key))
    {
        throw std::invalid_argument("JSON object does not contain required key: " + key);
    }
}

void to_json(nlohmann::json& j, const ScalarFeedbackMatrixOptions& config)
{
    j["matrix_size"] = config.matrix_size;
    j["type"] = config.type;
    if (config.custom_matrix.has_value())
    {
        j["custom_matrix"] = config.custom_matrix.value();
    }
    j["rng_seed"] = config.rng_seed;
    if (config.arg.has_value())
    {
        j["arg"] = config.arg.value();
    }
}

void from_json(const nlohmann::json& j, ScalarFeedbackMatrixOptions& config)
{
    config.matrix_size = j.at("matrix_size").get<uint32_t>();
    config.type = j.at("type").get<ScalarMatrixType>();
    if (j.contains("custom_matrix") && !j["custom_matrix"].is_null())
    {
        config.custom_matrix = j["custom_matrix"].get<std::vector<float>>();
    }
    config.rng_seed = j.at("rng_seed").get<uint32_t>();
    if (j.contains("arg") && !j["arg"].is_null())
    {
        config.arg = j["arg"].get<float>();
    }
}

void to_json(nlohmann::json& j, const DelayOptions& config)
{
    j["delay"] = config.delay;
    j["max_delay"] = config.max_delay;
    j["interp_type"] = config.interp_type;
    if (config.lfo_config.has_value())
    {
        j["lfo_config"] = config.lfo_config.value();
    }
}

void from_json(const nlohmann::json& j, DelayOptions& config)
{
    config.delay = j.at("delay").get<float>();
    config.max_delay = j.at("max_delay").get<uint32_t>();
    config.interp_type = j.at("interp_type").get<DelayInterpolationType>();
    if (j.contains("lfo_config") && !j["lfo_config"].is_null())
    {
        config.lfo_config = j["lfo_config"].get<ModulationOptions>();
    }
    else
    {
        config.lfo_config = std::nullopt;
    }
}

void to_json(nlohmann::json& j, const AttenuationFilterBankOptions& config)
{
    nlohmann::json filter_configs_json = nlohmann::json::array();
    for (const auto& filter_config : config.filter_configs)
    {
        filter_configs_json.push_back(std::visit(overloaded{[](const ProportionalAttenuationOptions& config) {
                                                                nlohmann::json j;
                                                                j["ProportionalAttenuationConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const TwoBandFilterOptions& config) {
                                                                nlohmann::json j;
                                                                j["TwoBandFilterConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const ThreeBandFilterOptions& config) {
                                                                nlohmann::json j;
                                                                j["ThreeBandFilterConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const TenBandFilterOptions& config) {
                                                                nlohmann::json j;
                                                                j["TenBandFilterConfig"] = config;
                                                                return j;
                                                            }},
                                                 filter_config));
    }
    j["AttenuationFilterBankOptions"] = filter_configs_json;
}

void from_json(const nlohmann::json& j, AttenuationFilterBankOptions& config)
{
    if (!j.is_array())
    {
        throw std::invalid_argument("AttenuationFilterBankOptions must be an array.");
    }

    config.filter_configs.clear();
    for (const auto& filter_config_json : j)
    {
        if (filter_config_json.contains("ProportionalAttenuationConfig"))
        {
            config.filter_configs.push_back(
                filter_config_json["ProportionalAttenuationConfig"].get<ProportionalAttenuationOptions>());
        }
        else if (filter_config_json.contains("TwoBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["TwoBandFilterConfig"].get<TwoBandFilterOptions>());
        }
        else if (filter_config_json.contains("ThreeBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["ThreeBandFilterConfig"].get<ThreeBandFilterOptions>());
        }
        else if (filter_config_json.contains("TenBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["TenBandFilterConfig"].get<TenBandFilterOptions>());
        }
        else
        {
            throw std::invalid_argument("Unknown filter config type in AttenuationFilterBankOptions");
        }
    }
}

nlohmann::json ToJson(const feedback_matrix_variant_t& matrix_config)
{
    return std::visit(overloaded{[](const CascadedFeedbackMatrixOptions& info) {
                                     nlohmann::json mat;
                                     mat["CascadedFeedbackMatrixInfo"] = info;
                                     return mat;
                                 },
                                 [](const ScalarFeedbackMatrixOptions& config) {
                                     nlohmann::json mat;
                                     mat["ScalarFeedbackMatrixOptions"] = config;
                                     return mat;
                                 }},
                      matrix_config);
}

nlohmann::json ToJson(const single_channel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{[](const SchroederAllpassSectionOptions& config) {
                                     nlohmann::json proc;
                                     proc["SchroederAllpassSectionOptions"] = config;
                                     return proc;
                                 },
                                 [](const AllpassFilterOptions& config) {
                                     nlohmann::json proc;
                                     proc["AllpassFilterOptions"] = config;
                                     return proc;
                                 },
                                 [](const CascadedBiquadsOptions& config) {
                                     nlohmann::json proc;
                                     proc["CascadedBiquadsOptions"] = config;
                                     return proc;
                                 },
                                 [](const FirOptions& config) {
                                     nlohmann::json proc;
                                     proc["FirOptions"] = config;
                                     return proc;
                                 },
                                 [](const DelayOptions& config) {
                                     nlohmann::json proc;
                                     proc["DelayOptions"] = config;
                                     return proc;
                                 },
                                 [](const GraphicEQOptions& config) {
                                     nlohmann::json proc;
                                     proc["GraphicEQOptions"] = config;
                                     return proc;
                                 }},
                      processor_config);
}

nlohmann::json ToJson(const multi_channel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{[](const ParallelGainsOptions& config) {
                                     nlohmann::json proc;
                                     proc["ParallelGainsConfig"] = config;
                                     return proc;
                                 },
                                 [](const ParallelSchroederAllpassSectionOptions& config) {
                                     nlohmann::json proc;
                                     proc["ParallelSchroederAllpassSectionOptions"] = config;
                                     return proc;
                                 },
                                 [](const AttenuationFilterBankOptions& config) {
                                     nlohmann::json proc = config;
                                     return proc;
                                 },
                                 [](const DelayBankOptions& config) {
                                     nlohmann::json proc;
                                     proc["DelayBankOptions"] = config;
                                     return proc;
                                 },
                                 [](const DelayBankTimeVaryingOptions& config) {
                                     nlohmann::json proc;
                                     proc["DelayBankTimeVaryingOptions"] = config;
                                     return proc;
                                 },
                                 [](const CascadedFeedbackMatrixOptions& config) {
                                     nlohmann::json proc;
                                     proc["CascadedFeedbackMatrixInfo"] = config;
                                     return proc;
                                 },
                                 [](const ScalarFeedbackMatrixOptions& config) {
                                     nlohmann::json proc;
                                     proc["ScalarFeedbackMatrixOptions"] = config;
                                     return proc;
                                 }},
                      processor_config);
}

single_channel_processor_variant_t SingleChannelProcessorFromJson(const nlohmann::json& j)
{
    if (j.contains("SchroederAllpassSectionOptions"))
    {
        return j["SchroederAllpassSectionOptions"].get<SchroederAllpassSectionOptions>();
    }
    else if (j.contains("AllpassFilterOptions"))
    {
        return j["AllpassFilterOptions"].get<AllpassFilterOptions>();
    }
    else if (j.contains("CascadedBiquadsOptions"))
    {
        return j["CascadedBiquadsOptions"].get<CascadedBiquadsOptions>();
    }
    else if (j.contains("FirOptions"))
    {
        return j["FirOptions"].get<FirOptions>();
    }
    else if (j.contains("DelayOptions"))
    {
        return j["DelayOptions"].get<DelayOptions>();
    }
    else if (j.contains("GraphicEQOptions"))
    {
        return j["GraphicEQOptions"].get<GraphicEQOptions>();
    }

    throw std::invalid_argument("Unknown single channel processor config type" + j.dump());
}

multi_channel_processor_variant_t MultichannelProcessorFromJson(const nlohmann::json& j)
{
    if (j.contains("ParallelGainsConfig"))
    {
        auto config = j["ParallelGainsConfig"].get<ParallelGainsOptions>();
        return config;
    }
    else if (j.contains("ParallelSchroederAllpassSectionOptions"))
    {
        return j["ParallelSchroederAllpassSectionOptions"].get<ParallelSchroederAllpassSectionOptions>();
    }
    else if (j.contains("AttenuationFilterBankOptions"))
    {
        auto config = j["AttenuationFilterBankOptions"].get<AttenuationFilterBankOptions>();
        return config;
    }
    else if (j.contains("DelayBankOptions"))
    {
        return j["DelayBankOptions"].get<DelayBankOptions>();
    }
    else if (j.contains("DelayBankTimeVaryingOptions"))
    {
        return j["DelayBankTimeVaryingOptions"].get<DelayBankTimeVaryingOptions>();
    }
    throw std::invalid_argument("Unknown multichannel processor config type");
}

feedback_matrix_variant_t FeedbackMatrixFromJson(const nlohmann::json& j)
{
    if (j.contains("CascadedFeedbackMatrixInfo"))
    {
        auto config = j["CascadedFeedbackMatrixInfo"].get<CascadedFeedbackMatrixOptions>();
        return config;
    }
    else if (j.contains("ScalarFeedbackMatrixOptions"))
    {
        ScalarFeedbackMatrixOptions config;
        from_json(j["ScalarFeedbackMatrixOptions"], config);
        // auto config = j["ScalarFeedbackMatrixOptions"].get<ScalarFeedbackMatrixOptions>();
        return config;
    }
    else
    {
        throw std::invalid_argument("Unknown feedback matrix config type");
    }
}

} // namespace sfFDN