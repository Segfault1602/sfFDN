#include "sffdn/serialization.h"

#include "json_helper.h"

#include <stdexcept>
#include <utility>

namespace sfFDN
{
void to_json(nlohmann::json& j, const InputStageConfig& p)
{
    nlohmann::json json;
    nlohmann::json single_channel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.single_channel_processors)
    {
        single_channel_processors_json.push_back(ToJson(processor_config));
    }
    json["single_channel_processors"] = single_channel_processors_json;
    json["parallel_gains_config"] = p.parallel_gains_config;
    nlohmann::json multichannel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.multichannel_processors)
    {
        multichannel_processors_json.push_back(ToJson(processor_config));
    }
    json["multichannel_processors"] = multichannel_processors_json;
    j = std::move(json);
}

void from_json(const nlohmann::json& j, InputStageConfig& p)
{
    json_detail::RequireObject(j, "InputStageConfig");
    if (j.size() != 3 || !j.contains("single_channel_processors") || !j.contains("parallel_gains_config") ||
        !j.contains("multichannel_processors"))
    {
        throw std::invalid_argument(
            "InputStageConfig must contain exactly single_channel_processors, parallel_gains_config, and "
            "multichannel_processors");
    }

    InputStageConfig candidate;
    candidate.parallel_gains_config = j.at("parallel_gains_config").get<StageGainsOptions>();
    const auto& single = j.at("single_channel_processors");
    const auto& multi = j.at("multichannel_processors");
    if (!single.is_array() || !multi.is_array())
    {
        throw std::invalid_argument("FDN processor lists must be arrays");
    }
    candidate.single_channel_processors.reserve(single.size());
    for (const auto& processor : single)
    {
        candidate.single_channel_processors.push_back(SingleChannelProcessorFromJson(processor));
    }
    candidate.multichannel_processors.reserve(multi.size());
    for (const auto& processor : multi)
    {
        candidate.multichannel_processors.push_back(MultichannelProcessorFromJson(processor));
    }
    p = std::move(candidate);
}

void to_json(nlohmann::json& j, const OutputStageConfig& p)
{
    nlohmann::json json;
    nlohmann::json multichannel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.multichannel_processors)
    {
        multichannel_processors_json.push_back(ToJson(processor_config));
    }
    json["multichannel_processors"] = multichannel_processors_json;
    json["parallel_gains_config"] = p.parallel_gains_config;
    nlohmann::json single_channel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.single_channel_processors)
    {
        single_channel_processors_json.push_back(ToJson(processor_config));
    }
    json["single_channel_processors"] = single_channel_processors_json;
    j = std::move(json);
}

void from_json(const nlohmann::json& j, OutputStageConfig& p)
{
    json_detail::RequireObject(j, "OutputStageConfig");
    if (j.size() != 3 || !j.contains("multichannel_processors") || !j.contains("parallel_gains_config") ||
        !j.contains("single_channel_processors"))
    {
        throw std::invalid_argument(
            "OutputStageConfig must contain exactly multichannel_processors, parallel_gains_config, and "
            "single_channel_processors");
    }

    OutputStageConfig candidate;
    candidate.parallel_gains_config = j.at("parallel_gains_config").get<StageGainsOptions>();
    const auto& multi = j.at("multichannel_processors");
    const auto& single = j.at("single_channel_processors");
    if (!multi.is_array() || !single.is_array())
    {
        throw std::invalid_argument("FDN processor lists must be arrays");
    }
    candidate.multichannel_processors.reserve(multi.size());
    for (const auto& processor : multi)
    {
        candidate.multichannel_processors.push_back(MultichannelProcessorFromJson(processor));
    }
    candidate.single_channel_processors.reserve(single.size());
    for (const auto& processor : single)
    {
        candidate.single_channel_processors.push_back(SingleChannelProcessorFromJson(processor));
    }
    p = std::move(candidate);
}

void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p)
{
    nlohmann::json json;
    json["fdn_size"] = p.fdn_size;
    json["transposed"] = p.transposed;
    json["direct_gain"] = p.direct_gain;
    json["block_size"] = p.block_size;
    json["sample_rate"] = p.sample_rate;
    json["delay_bank_config"] = p.delay_bank_config;
    json["input_block_config"] = p.input_block_config;

    json["feedback_matrix_config"] = ToJson(p.feedback_matrix_config);

    json["attenuation_filter_bank_config"] =
        p.attenuation_filter_bank_config.has_value() ? ToJson(p.attenuation_filter_bank_config.value()) : nullptr;

    nlohmann::json loop_filter_configs_json = nlohmann::json::array();
    for (const auto& processor_config : p.loop_filter_configs)
    {
        loop_filter_configs_json.push_back(ToJson(processor_config));
    }
    json["loop_filter_configs"] = loop_filter_configs_json;

    json["output_block_config"] = p.output_block_config;

    json["tone_correction_filters"] = nlohmann::json::array();
    for (const auto& processor_config : p.tone_correction_filters)
    {
        json["tone_correction_filters"].push_back(ToJson(processor_config));
    }

    j = json;
}

void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p)
{
    json_detail::RequireObject(j, "FDNConfig");
    FDNConfig candidate;
    json_detail::ReadField(j, "fdn_size", candidate.fdn_size);
    json_detail::ReadField(j, "transposed", candidate.transposed);
    json_detail::ReadField(j, "direct_gain", candidate.direct_gain);
    json_detail::ReadField(j, "block_size", candidate.block_size);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    json_detail::ReadField(j, "delay_bank_config", candidate.delay_bank_config);

    candidate.input_block_config = j.at("input_block_config").get<InputStageConfig>();
    candidate.feedback_matrix_config = FeedbackMatrixFromJson(j.at("feedback_matrix_config"));

    const auto& attenuation = j.at("attenuation_filter_bank_config");
    if (!attenuation.is_null())
    {
        if (!attenuation.is_object() || attenuation.size() != 1 ||
            !attenuation.contains("AttenuationFilterBankOptions"))
        {
            throw std::invalid_argument("Attenuation filter bank config must contain exactly one filter type");
        }
        candidate.attenuation_filter_bank_config =
            attenuation.at("AttenuationFilterBankOptions").get<AttenuationFilterBankOptions>();
    }

    const auto& loop = j.at("loop_filter_configs");
    if (!loop.is_array())
    {
        throw std::invalid_argument("FDN loop filter configs must be an array");
    }
    candidate.loop_filter_configs.reserve(loop.size());
    for (const auto& processor : loop)
    {
        candidate.loop_filter_configs.push_back(MultichannelProcessorFromJson(processor));
    }

    candidate.output_block_config = j.at("output_block_config").get<OutputStageConfig>();

    const auto& tone = j.at("tone_correction_filters");
    if (!tone.is_array())
    {
        throw std::invalid_argument("FDN tone correction filters must be an array");
    }
    candidate.tone_correction_filters.reserve(tone.size());
    for (const auto& processor : tone)
    {
        candidate.tone_correction_filters.push_back(SingleChannelProcessorFromJson(processor));
    }
    p = std::move(candidate);
}

} // namespace sfFDN
