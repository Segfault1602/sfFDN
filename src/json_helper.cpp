#include "json_helper.h"

#include "sffdn/types.h"

#include <cstdint>
#include <map>

namespace sfFDN
{

using AudioProcessorFactoryFunction = std::function<std::unique_ptr<AudioProcessor>(const nlohmann::json&)>;

static const std::map<std::string, AudioProcessorFactoryFunction> processor_factory = {
    {"AudioProcessorChain", AudioProcessorChain::FromJson},
    {"DelayBank", DelayBank::FromJson},
    {"ParallelGains", ParallelGains::FromJson},
    {"FilterBank", FilterBank::FromJson},
    {"IIRFilterBank", IIRFilterBank::FromJson},
    {"CascadedBiquads", CascadedBiquads::FromJson},
    {"ScalarFeedbackMatrix", ScalarFeedbackMatrix::FromJson},
    {"FilterFeedbackMatrix", FilterFeedbackMatrix::FromJson},
    {"TimeVaryingParallelGains", TimeVaryingParallelGains::FromJson},
    {"SchroederAllpassSection", SchroederAllpassSection::FromJson},
    {"ParallelSchroederAllpassSection", ParallelSchroederAllpassSection::FromJson}};

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

std::unique_ptr<AudioProcessor> from_json(const nlohmann::json& j)
{
    if (!j.contains("type") || !j["type"].is_string())
    {
        throw std::runtime_error("JSON object does not contain a valid 'type' field.");
    }

    std::string type = j["type"].get<std::string>();

    auto it = processor_factory.find(type);
    if (it == processor_factory.end())
    {
        throw std::runtime_error("Unknown audio processor type: " + type);
    }

    return it->second(j);
}

void to_json(nlohmann::json& j, const ScalarFeedbackMatrixConfig& config)
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

void from_json(const nlohmann::json& j, ScalarFeedbackMatrixConfig& config)
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

void to_json(nlohmann::json& j, const DelayConfig& config)
{
    j["delay"] = config.delay;
    j["max_delay"] = config.max_delay;
    j["interp_type"] = config.interp_type;
    if (config.lfo_config.has_value())
    {
        j["lfo_config"] = config.lfo_config.value();
    }
}

void from_json(const nlohmann::json& j, DelayConfig& config)
{
    config.delay = j.at("delay").get<float>();
    config.max_delay = j.at("max_delay").get<uint32_t>();
    config.interp_type = j.at("interp_type").get<DelayInterpolationType>();
    if (j.contains("lfo_config") && !j["lfo_config"].is_null())
    {
        config.lfo_config = j["lfo_config"].get<ModulationConfig>();
    }
    else
    {
        config.lfo_config = std::nullopt;
    }
}

void to_json(nlohmann::json& j, const AttenuationFilterBankConfig& config)
{
    nlohmann::json filter_configs_json = nlohmann::json::array();
    for (const auto& filter_config : config.filter_configs)
    {
        filter_configs_json.push_back(std::visit(overloaded{[](const ProportionalAttenuationConfig& config) {
                                                                nlohmann::json j;
                                                                j["ProportionalAttenuationConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const TwoBandFilterConfig& config) {
                                                                nlohmann::json j;
                                                                j["TwoBandFilterConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const ThreeBandFilterConfig& config) {
                                                                nlohmann::json j;
                                                                j["ThreeBandFilterConfig"] = config;
                                                                return j;
                                                            },
                                                            [](const TenBandFilterConfig& config) {
                                                                nlohmann::json j;
                                                                j["TenBandFilterConfig"] = config;
                                                                return j;
                                                            }},
                                                 filter_config));
    }
    j["AttenuationFilterBankConfig"] = filter_configs_json;
}

void from_json(const nlohmann::json& j, AttenuationFilterBankConfig& config)
{
    if (!j.is_array())
    {
        throw std::invalid_argument("AttenuationFilterBankConfig must be an array.");
    }

    config.filter_configs.clear();
    for (const auto& filter_config_json : j)
    {
        if (filter_config_json.contains("ProportionalAttenuationConfig"))
        {
            config.filter_configs.push_back(
                filter_config_json["ProportionalAttenuationConfig"].get<ProportionalAttenuationConfig>());
        }
        else if (filter_config_json.contains("TwoBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["TwoBandFilterConfig"].get<TwoBandFilterConfig>());
        }
        else if (filter_config_json.contains("ThreeBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["ThreeBandFilterConfig"].get<ThreeBandFilterConfig>());
        }
        else if (filter_config_json.contains("TenBandFilterConfig"))
        {
            config.filter_configs.push_back(filter_config_json["TenBandFilterConfig"].get<TenBandFilterConfig>());
        }
        else
        {
            throw std::invalid_argument("Unknown filter config type in AttenuationFilterBankConfig");
        }
    }
}

nlohmann::json ToJson(const feedback_matrix_variant_t& matrix_config)
{
    return std::visit(overloaded{[](const CascadedFeedbackMatrixInfo& info) {
                                     nlohmann::json mat;
                                     mat["CascadedFeedbackMatrixInfo"] = info;
                                     return mat;
                                 },
                                 [](const ScalarFeedbackMatrixConfig& config) {
                                     nlohmann::json mat;
                                     mat["ScalarFeedbackMatrixConfig"] = config;
                                     return mat;
                                 }},
                      matrix_config);
}

nlohmann::json ToJson(const single_channel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{[](const SchroederAllpassSectionConfig& config) {
                                     nlohmann::json proc;
                                     proc["SchroederAllpassSectionConfig"] = config;
                                     return proc;
                                 },
                                 [](const AllpassFilterConfig& config) {
                                     nlohmann::json proc;
                                     proc["AllpassFilterConfig"] = config;
                                     return proc;
                                 },
                                 [](const CascadedBiquadsConfig& config) {
                                     nlohmann::json proc;
                                     proc["CascadedBiquadsConfig"] = config;
                                     return proc;
                                 },
                                 [](const FirConfig& config) {
                                     nlohmann::json proc;
                                     proc["FirConfig"] = config;
                                     return proc;
                                 },
                                 [](const DelayConfig& config) {
                                     nlohmann::json proc;
                                     proc["DelayConfig"] = config;
                                     return proc;
                                 },
                                 [](const GraphicEQConfig& config) {
                                     nlohmann::json proc;
                                     proc["GraphicEQConfig"] = config;
                                     return proc;
                                 }},
                      processor_config);
}

nlohmann::json ToJson(const multichannel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{[](const ParallelGainsConfig& config) {
                                     nlohmann::json proc;
                                     proc["ParallelGainsConfig"] = config;
                                     return proc;
                                 },
                                 [](const ParallelSchroederAllpassSectionConfig& config) {
                                     nlohmann::json proc;
                                     proc["ParallelSchroederAllpassSectionConfig"] = config;
                                     return proc;
                                 },
                                 [](const AttenuationFilterBankConfig& config) {
                                     nlohmann::json proc = config;
                                     return proc;
                                 },
                                 [](const DelayBankConfig& config) {
                                     nlohmann::json proc;
                                     proc["DelayBankConfig"] = config;
                                     return proc;
                                 },
                                 [](const DelayBankTimeVaryingConfig& config) {
                                     nlohmann::json proc;
                                     proc["DelayBankTimeVaryingConfig"] = config;
                                     return proc;
                                 },
                                 [](const CascadedFeedbackMatrixInfo& config) {
                                     nlohmann::json proc;
                                     proc["CascadedFeedbackMatrixInfo"] = config;
                                     return proc;
                                 },
                                 [](const ScalarFeedbackMatrixConfig& config) {
                                     nlohmann::json proc;
                                     proc["ScalarFeedbackMatrixConfig"] = config;
                                     return proc;
                                 }},
                      processor_config);
}

single_channel_processor_variant_t SingleChannelProcessorFromJson(const nlohmann::json& j)
{
    if (j.contains("SchroederAllpassSectionConfig"))
    {
        return j["SchroederAllpassSectionConfig"].get<SchroederAllpassSectionConfig>();
    }
    else if (j.contains("AllpassFilterConfig"))
    {
        return j["AllpassFilterConfig"].get<AllpassFilterConfig>();
    }
    else if (j.contains("CascadedBiquadsConfig"))
    {
        return j["CascadedBiquadsConfig"].get<CascadedBiquadsConfig>();
    }
    else if (j.contains("FirConfig"))
    {
        return j["FirConfig"].get<FirConfig>();
    }
    else if (j.contains("DelayConfig"))
    {
        return j["DelayConfig"].get<DelayConfig>();
    }
    else if (j.contains("GraphicEQConfig"))
    {
        return j["GraphicEQConfig"].get<GraphicEQConfig>();
    }

    throw std::invalid_argument("Unknown single channel processor config type" + j.dump());
}

multichannel_processor_variant_t MultichannelProcessorFromJson(const nlohmann::json& j)
{
    if (j.contains("ParallelGainsConfig"))
    {
        auto config = j["ParallelGainsConfig"].get<ParallelGainsConfig>();
        return config;
    }
    else if (j.contains("ParallelSchroederAllpassSectionConfig"))
    {
        return j["ParallelSchroederAllpassSectionConfig"].get<ParallelSchroederAllpassSectionConfig>();
    }
    else if (j.contains("AttenuationFilterBankConfig"))
    {
        auto config = j["AttenuationFilterBankConfig"].get<AttenuationFilterBankConfig>();
        return config;
    }
    else if (j.contains("DelayBankConfig"))
    {
        return j["DelayBankConfig"].get<DelayBankConfig>();
    }
    else if (j.contains("DelayBankTimeVaryingConfig"))
    {
        return j["DelayBankTimeVaryingConfig"].get<DelayBankTimeVaryingConfig>();
    }
    throw std::invalid_argument("Unknown multichannel processor config type");
}

feedback_matrix_variant_t FeedbackMatrixFromJson(const nlohmann::json& j)
{
    if (j.contains("CascadedFeedbackMatrixInfo"))
    {
        auto config = j["CascadedFeedbackMatrixInfo"].get<CascadedFeedbackMatrixInfo>();
        return config;
    }
    else if (j.contains("ScalarFeedbackMatrixConfig"))
    {
        ScalarFeedbackMatrixConfig config;
        from_json(j["ScalarFeedbackMatrixConfig"], config);
        // auto config = j["ScalarFeedbackMatrixConfig"].get<ScalarFeedbackMatrixConfig>();
        return config;
    }
    else
    {
        throw std::invalid_argument("Unknown feedback matrix config type");
    }
}

} // namespace sfFDN