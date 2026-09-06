#include "sffdn/fdn_config.h"

#include "json_helper.h"
#include "processor_factory.h"

#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/filter_design.h"
#include "sffdn/filter_feedback_matrix.h"
#include "sffdn/filterbank.h"
#include "sffdn/parallel_gains.h"
#include "sffdn/time_varying_feedback_matrix.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace
{
struct MultichannelProcessorVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ParallelGainsOptions& gains_config) const
    {
        return MakeParallelGainsFromConfig(gains_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::MultichannelProcessorOptions& config) const
    {
        return std::make_unique<sfFDN::FilterBank>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::AttenuationFilterBankOptions& attenuation_config) const
    {
        return sfFDN::CreateAttenuationFilterBank(attenuation_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayBankOptions& delay_bank_config) const
    {
        return std::make_unique<sfFDN::DelayBank>(delay_bank_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayBankTimeVaryingOptions& delay_bank_config) const
    {
        return std::make_unique<sfFDN::DelayBankTimeVarying>(delay_bank_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedFeedbackMatrixOptions& matrix_config) const
    {
        return std::make_unique<sfFDN::FilterFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ScalarFeedbackMatrixOptions& matrix_config) const
    {
        return std::make_unique<sfFDN::ScalarFeedbackMatrix>(matrix_config);
    }
};

void AddProcessorOrThrow(sfFDN::AudioProcessorChain& chain, std::unique_ptr<sfFDN::AudioProcessor> processor,
                         const char* context)
{
    if (!chain.AddProcessor(std::move(processor)))
    {
        throw std::runtime_error(std::string("Failed to add ") + context + " to audio processor chain");
    }
}

std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const sfFDN::FDNConfig& config)
{
    std::unique_ptr<sfFDN::AudioProcessor> input_gains =
        MakeParallelGainsFromConfig(config.input_block_config.parallel_gains_config);

    if (config.input_block_config.single_channel_processors.empty() &&
        config.input_block_config.multichannel_processors.empty())
    {
        return input_gains;
    }

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(config.block_size);

    for (const auto& processor_config : config.input_block_config.single_channel_processors)
    {
        auto processor = sfFDN::CreateSingleChannelProcessor(processor_config);
        AddProcessorOrThrow(*chain_processor, std::move(processor), "input single-channel processor");
    }

    AddProcessorOrThrow(*chain_processor, std::move(input_gains), "input gains");
    for (const auto& processor_config : config.input_block_config.multichannel_processors)
    {
        auto processor = std::visit(MultichannelProcessorVisitor{}, processor_config);
        AddProcessorOrThrow(*chain_processor, std::move(processor), "input multichannel processor");
    }

    return chain_processor;
}

std::unique_ptr<sfFDN::AudioProcessor> CreateOutputGainsFromConfig(const sfFDN::FDNConfig& config)
{
    std::unique_ptr<sfFDN::AudioProcessor> output_gains =
        MakeParallelGainsFromConfig(config.output_block_config.parallel_gains_config);

    if (config.output_block_config.single_channel_processors.empty() &&
        config.output_block_config.multichannel_processors.empty())
    {
        return output_gains;
    }

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(config.block_size);

    for (const auto& processor_config : config.output_block_config.multichannel_processors)
    {
        AddProcessorOrThrow(*chain_processor, std::visit(MultichannelProcessorVisitor{}, processor_config),
                            "output multichannel processor");
    }

    AddProcessorOrThrow(*chain_processor, std::move(output_gains), "output gains");

    for (const auto& processor_config : config.output_block_config.single_channel_processors)
    {
        AddProcessorOrThrow(*chain_processor, sfFDN::CreateSingleChannelProcessor(processor_config),
                            "output single-channel processor");
    }

    return chain_processor;
}

struct FeedbackMatrixVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedFeedbackMatrixOptions& matrix_config) const
    {
        return std::make_unique<sfFDN::FilterFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ScalarFeedbackMatrixOptions& matrix_config) const
    {
        return std::make_unique<sfFDN::ScalarFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::TimeVaryingFeedbackMatrixOptions& matrix_config) const
    {
        return std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const std::vector<float>& matrix_config) const
    {
        const auto matrix_size = static_cast<uint32_t>(std::sqrt(matrix_config.size()));

        if (matrix_size * matrix_size != matrix_config.size())
        {
            throw std::runtime_error("Custom scalar feedback matrix size must be a perfect square");
        }

        sfFDN::ScalarFeedbackMatrixOptions scalar_config;
        scalar_config.matrix_size = matrix_size;
        scalar_config.custom_matrix = matrix_config;
        return std::make_unique<sfFDN::ScalarFeedbackMatrix>(scalar_config);
    }
};

sfFDN::multi_channel_processor_variant_t UpdateAttenuationFilterBank(
    const sfFDN::multi_channel_processor_variant_t& processor_config, const sfFDN::FDNConfig& config)
{
    if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(processor_config))
    {
        const auto& attenuation_config = std::get<sfFDN::AttenuationFilterBankOptions>(processor_config);
        sfFDN::AttenuationFilterBankOptions updated_config = attenuation_config;
        // Always update the delays in the attenuation filter bank to match the current delay lengths
        if (attenuation_config.filter_configs.size() != config.fdn_size)
        {
            auto filter_config = attenuation_config.filter_configs.back();
            std::visit(sfFDN::overloaded{[&](auto& arg) { arg.delay = 0.f; }}, filter_config);
            updated_config.filter_configs.clear();

            // Copy the last filter config to match the number of channels in the FDN
            for (size_t i = 0; i < config.fdn_size; ++i)
            {
                updated_config.filter_configs.emplace_back(filter_config);
            }
        }

        for (size_t i = 0; i < config.fdn_size; ++i)
        {
            auto& filter_config = updated_config.filter_configs[i];
            std::visit(sfFDN::overloaded{[&](auto& arg) {
                           if (arg.delay <= 0.f)
                           {
                               arg.delay = config.delay_bank_config.delays[i];
                           }
                       }},
                       filter_config);
        }
        return updated_config;
    }

    return processor_config;
}

} // namespace

namespace sfFDN
{
std::unique_ptr<FDN> CreateFDNFromConfig(const FDNConfig& config)
{
    auto validation = ValidateFDNStructure(config);
    if (!validation.has_value())
    {
        throw FDNConfigError(std::move(validation.error()));
    }
    auto fdn = std::make_unique<FDN>(config.fdn_size, config.block_size);
    fdn->SetTranspose(config.transposed);
    fdn->SetDirectGain(config.direct_gain);

    // Delaybank
    if (!fdn->SetDelayBank(config.delay_bank_config))
    {
        throw std::runtime_error("Failed to set FDN delay bank");
    }

    // Input gain Block
    if (!fdn->SetInputGains(CreateInputGainsFromConfig(config)))
    {
        throw std::runtime_error("Failed to set FDN input gains");
    }

    // Feedback matrix block
    try
    {
        if (!fdn->SetFeedbackMatrix(std::visit(FeedbackMatrixVisitor{}, config.feedback_matrix_config)))
        {
            throw std::runtime_error("Failed to set FDN feedback matrix");
        }
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(std::string("Invalid feedback matrix configuration: ") + error.what());
    }

    std::unique_ptr<AudioProcessor> attenuation_filter_bank = nullptr;
    if (config.attenuation_filter_bank_config.has_value())
    {
        attenuation_filter_bank =
            std::visit(MultichannelProcessorVisitor{},
                       UpdateAttenuationFilterBank(config.attenuation_filter_bank_config.value(), config));
    }

    // Loop filter block
    if (!config.loop_filter_configs.empty())
    {
        if (config.loop_filter_configs.size() == 1 && attenuation_filter_bank == nullptr)
        {
            auto updated_config = UpdateAttenuationFilterBank(config.loop_filter_configs[0], config);
            auto processor = std::visit(MultichannelProcessorVisitor{}, updated_config);
            if (!fdn->SetLoopFilter(std::move(processor)))
            {
                throw std::runtime_error("Failed to set FDN loop filter");
            }
        }
        else if (!config.loop_filter_configs.empty())
        {
            auto loop_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);

            if (attenuation_filter_bank != nullptr)
            {
                AddProcessorOrThrow(*loop_filter_chain, std::move(attenuation_filter_bank), "attenuation filter bank");
            }

            for (const auto& processor_config : config.loop_filter_configs)
            {
                auto updated_config = UpdateAttenuationFilterBank(processor_config, config);
                auto processor = std::visit(MultichannelProcessorVisitor{}, updated_config);
                AddProcessorOrThrow(*loop_filter_chain, std::move(processor), "loop filter");
            }
            if (!fdn->SetLoopFilter(std::move(loop_filter_chain)))
            {
                throw std::runtime_error("Failed to set FDN loop filter");
            }
        }
    }
    else
    {
        if (!fdn->SetLoopFilter(std::move(attenuation_filter_bank)))
        {
            throw std::runtime_error("Failed to set FDN loop filter");
        }
    }

    // TC filters
    if (!config.tone_correction_filters.empty())
    {
        if (config.tone_correction_filters.size() == 1)
        {
            auto processor = CreateSingleChannelProcessor(config.tone_correction_filters[0]);
            if (!fdn->SetTCFilter(std::move(processor)))
            {
                throw std::runtime_error("Failed to set FDN tone correction filter");
            }
        }
        else
        {
            auto tc_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);
            for (const auto& processor_config : config.tone_correction_filters)
            {
                auto processor = CreateSingleChannelProcessor(processor_config);
                AddProcessorOrThrow(*tc_filter_chain, std::move(processor), "tone correction filter");
            }
            if (!fdn->SetTCFilter(std::move(tc_filter_chain)))
            {
                throw std::runtime_error("Failed to set FDN tone correction filter");
            }
        }
    }

    // Output gain block
    if (!fdn->SetOutputGains(CreateOutputGainsFromConfig(config)))
    {
        throw std::runtime_error("Failed to set FDN output gains");
    }

    return fdn;
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

    nlohmann::json input_block_json;
    nlohmann::json single_channel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.input_block_config.single_channel_processors)
    {
        single_channel_processors_json.push_back(ToJson(processor_config));
    }
    input_block_json["single_channel_processors"] = single_channel_processors_json;
    input_block_json["parallel_gains_config"] = p.input_block_config.parallel_gains_config;
    nlohmann::json multichannel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.input_block_config.multichannel_processors)
    {
        multichannel_processors_json.push_back(ToJson(processor_config));
    }
    input_block_json["multichannel_processors"] = multichannel_processors_json;
    json["input_block_config"] = input_block_json;

    json["feedback_matrix_config"] = ToJson(p.feedback_matrix_config);

    json["attenuation_filter_bank_config"] =
        p.attenuation_filter_bank_config.has_value() ? ToJson(p.attenuation_filter_bank_config.value()) : nullptr;

    nlohmann::json loop_filter_configs_json = nlohmann::json::array();
    for (const auto& processor_config : p.loop_filter_configs)
    {
        loop_filter_configs_json.push_back(ToJson(processor_config));
    }
    json["loop_filter_configs"] = loop_filter_configs_json;

    nlohmann::json output_block_json;
    nlohmann::json output_single_channel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.output_block_config.single_channel_processors)
    {
        output_single_channel_processors_json.push_back(ToJson(processor_config));
    }
    output_block_json["single_channel_processors"] = output_single_channel_processors_json;
    output_block_json["parallel_gains_config"] = p.output_block_config.parallel_gains_config;
    nlohmann::json output_multichannel_processors_json = nlohmann::json::array();
    for (const auto& processor_config : p.output_block_config.multichannel_processors)
    {
        output_multichannel_processors_json.push_back(ToJson(processor_config));
    }
    output_block_json["multichannel_processors"] = output_multichannel_processors_json;
    json["output_block_config"] = output_block_json;

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

    const auto parse_block = [](const nlohmann::json& block, auto& destination) {
        json_detail::RequireObject(block, "FDN block config");
        destination.parallel_gains_config = block.at("parallel_gains_config").get<ParallelGainsOptions>();
        const auto& single = block.at("single_channel_processors");
        const auto& multi = block.at("multichannel_processors");
        if (!single.is_array() || !multi.is_array())
        {
            throw std::invalid_argument("FDN processor lists must be arrays");
        }
        destination.single_channel_processors.reserve(single.size());
        for (const auto& processor : single)
        {
            destination.single_channel_processors.push_back(SingleChannelProcessorFromJson(processor));
        }
        destination.multichannel_processors.reserve(multi.size());
        for (const auto& processor : multi)
        {
            destination.multichannel_processors.push_back(MultichannelProcessorFromJson(processor));
        }
    };

    parse_block(j.at("input_block_config"), candidate.input_block_config);
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

    parse_block(j.at("output_block_config"), candidate.output_block_config);

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
