#include "json_helper.h"

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
} // namespace sfFDN