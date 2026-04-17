#include "sffdn/fdn_config2.h"

#include "json_helper.h"
#include "sffdn/sffdn.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <variant>

namespace
{
bool ValidateConfig(const sfFDN::FDNConfig2& config)
{
    if (config.fdn_size == 0)
    {
        std::cerr << "FDN size must be greater than 0" << std::endl;
        return false;
    }

    if (config.fdn_size != config.delay_bank_config.delays.size())
    {
        std::cerr << "Number of delays in delay bank config must match FDN size" << std::endl;
        return false;
    }

    if (config.fdn_size != config.input_block_config.parallel_gains_config.gains.size())
    {
        std::cerr << "Number of gains in input parallel gains config must match FDN size" << std::endl;
        return false;
    }

    if (config.fdn_size != config.output_block_config.parallel_gains_config.gains.size())
    {
        std::cerr << "Number of gains in output parallel gains config must match FDN size" << std::endl;
        return false;
    }

    bool feedback_matrix_valid = std::visit(
        [&](const auto& matrix_config) -> bool {
            if (matrix_config.matrix_size != config.fdn_size)
            {
                std::cerr << "Feedback matrix channel count must match FDN size" << std::endl;
                return false;
            }

            return true;
        },
        config.feedback_matrix_config);

    if (!feedback_matrix_valid)
    {
        return false;
    }

    return true;
}

struct SingleChannelProcessorVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::SchroederAllpassSectionOptions& config) const
    {
        return std::make_unique<sfFDN::SchroederAllpassSection>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::AllpassFilterOptions& config) const
    {
        auto filter = std::make_unique<sfFDN::AllpassFilter>();
        filter->SetCoefficients(config.coeff);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedBiquadsOptions& config) const
    {
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(config.coeffs);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::FirOptions& config) const
    {
        auto filter = sfFDN::MakeFirFilter(config);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayOptions& config) const
    {
        if (config.lfo_config.has_value())
        {
            return std::make_unique<sfFDN::DelayTimeVarying>(config);
        }

        return std::make_unique<sfFDN::DelayInterp>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::GraphicEQOptions& config) const
    {
        auto sos = sfFDN::DesignGraphicEQ(config);
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(sos);
        return filter;
    }
};

struct MultichannelProcessorVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ParallelGainsOptions& gains_config) const
    {
        return MakeParallelGainsFromConfig(gains_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::ParallelSchroederAllpassSectionOptions& schroeder_config) const
    {
        auto bank = std::make_unique<sfFDN::FilterBank>();
        for (const auto& section_config : schroeder_config.sections)
        {
            auto schroeder = std::make_unique<sfFDN::SchroederAllpassSection>(section_config);
            bank->AddFilter(std::move(schroeder));
        }
        return bank;
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

std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const sfFDN::FDNConfig2& config)
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
        auto processor = std::visit(SingleChannelProcessorVisitor{}, processor_config);
        chain_processor->AddProcessor(std::move(processor));
    }

    chain_processor->AddProcessor(std::move(input_gains));
    for (const auto& processor_config : config.input_block_config.multichannel_processors)
    {
        auto processor = std::visit(MultichannelProcessorVisitor{}, processor_config);
        chain_processor->AddProcessor(std::move(processor));
    }

    return chain_processor;
}

std::unique_ptr<sfFDN::AudioProcessor> CreateOutputGainsFromConfig(const sfFDN::FDNConfig2& config)
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
        chain_processor->AddProcessor(std::visit(MultichannelProcessorVisitor{}, processor_config));
    }

    chain_processor->AddProcessor(std::move(output_gains));

    for (const auto& processor_config : config.output_block_config.single_channel_processors)
    {
        chain_processor->AddProcessor(std::visit(SingleChannelProcessorVisitor{}, processor_config));
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

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const std::vector<float>& matrix_config) const
    {
        uint32_t matrix_size = static_cast<uint32_t>(std::sqrt(matrix_config.size()));

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
    const sfFDN::multi_channel_processor_variant_t& processor_config, const sfFDN::FDNConfig2& config)
{
    if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(processor_config))
    {
        const auto& attenuation_config = std::get<sfFDN::AttenuationFilterBankOptions>(processor_config);
        sfFDN::AttenuationFilterBankOptions updated_config = attenuation_config;
        // Always update the delays in the attenuation filter bank to match the current delay lengths
        if (attenuation_config.filter_configs.size() != config.fdn_size)
        {
            auto filter_config = attenuation_config.filter_configs.back();

            // Copy the last filter config to match the number of channels in the FDN
            for (size_t i = attenuation_config.filter_configs.size(); i < config.fdn_size; ++i)
            {
                updated_config.filter_configs.push_back(filter_config);
            }
        }

        for (size_t i = 0; i < config.fdn_size; ++i)
        {
            auto& filter_config = updated_config.filter_configs[i];
            std::visit(sfFDN::overloaded{[&](auto& arg) { arg.delay = config.delay_bank_config.delays[i]; }},
                       filter_config);
        }
        return updated_config;
    }

    return processor_config;
}

} // namespace

namespace sfFDN
{
std::unique_ptr<FDN> CreateFDNFromConfig2(const FDNConfig2& config)
{
    if (!ValidateConfig(config))
    {
        throw std::runtime_error("Invalid FDNConfig2");
    }
    auto fdn = std::make_unique<FDN>(config.fdn_size, config.block_size);
    fdn->SetTranspose(config.transposed);
    fdn->SetDirectGain(config.direct_gain);

    // Delaybank
    fdn->SetDelayBank(config.delay_bank_config);

    // Input gain Block
    fdn->SetInputGains(CreateInputGainsFromConfig(config));

    // Feedback matrix block
    fdn->SetFeedbackMatrix(std::visit(FeedbackMatrixVisitor{}, config.feedback_matrix_config));

    // Loop filter block
    if (!config.loop_filter_configs.empty())
    {
        if (config.loop_filter_configs.size() > 1)
        {
            auto loop_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);
            for (const auto& processor_config : config.loop_filter_configs)
            {
                auto updated_config = UpdateAttenuationFilterBank(processor_config, config);
                auto processor = std::visit(MultichannelProcessorVisitor{}, updated_config);
                loop_filter_chain->AddProcessor(std::move(processor));
            }
            fdn->SetFilterBank(std::move(loop_filter_chain));
        }
        else
        {
            auto updated_config = UpdateAttenuationFilterBank(config.loop_filter_configs[0], config);
            auto processor = std::visit(MultichannelProcessorVisitor{}, updated_config);
            fdn->SetFilterBank(std::move(processor));
        }
    }

    // Output gain block
    fdn->SetOutputGains(CreateOutputGainsFromConfig(config));

    return fdn;
}

template <typename T>
std::string VariantTypeName()
{
    if constexpr (std::is_same_v<T, ParallelGainsOptions>)
    {
        return "ParallelGainsOptions";
    }
    else if constexpr (std::is_same_v<T, ParallelSchroederAllpassSectionOptions>)
    {
        return "ParallelSchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, AttenuationFilterBankOptions>)
    {
        return "AttenuationFilterBankOptions";
    }
    else if constexpr (std::is_same_v<T, SchroederAllpassSectionOptions>)
    {
        return "SchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, AllpassFilterOptions>)
    {
        return "AllpassFilterOptions";
    }
    else if constexpr (std::is_same_v<T, CascadedBiquadsOptions>)
    {
        return "CascadedBiquadsOptions";
    }
    else if constexpr (std::is_same_v<T, FirOptions>)
    {
        return "FirOptions";
    }
    else if constexpr (std::is_same_v<T, DelayOptions>)
    {
        return "DelayOptions";
    }
    else if constexpr (std::is_same_v<T, DelayBankOptions>)
    {
        return "DelayBankOptions";
    }
    else if constexpr (std::is_same_v<T, DelayBankTimeVaryingOptions>)
    {
        return "DelayBankTimeVaryingOptions";
    }
    else if constexpr (std::is_same_v<T, CascadedFeedbackMatrixOptions>)
    {
        return "CascadedFeedbackMatrixInfo";
    }
    else if constexpr (std::is_same_v<T, ScalarFeedbackMatrixOptions>)
    {
        return "ScalarFeedbackMatrixOptions";
    }
    else
    {
        throw std::runtime_error("Unsupported variant type");
    }
}

void to_json(nlohmann::json& j, const sfFDN::FDNConfig2& p)
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
        std::visit(
            [&](const auto& config) {
                single_channel_processors_json.push_back({{VariantTypeName<std::decay_t<decltype(config)>>(), config}});
            },
            processor_config);
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

void from_json(const nlohmann::json& j, sfFDN::FDNConfig2& p)
{
    p.fdn_size = j.at("fdn_size").get<uint32_t>();
    p.transposed = j.at("transposed").get<bool>();
    p.direct_gain = j.at("direct_gain").get<float>();
    p.block_size = j.at("block_size").get<uint32_t>();
    p.sample_rate = j.at("sample_rate").get<uint32_t>();
    p.delay_bank_config = j.at("delay_bank_config").get<DelayBankOptions>();

    const auto& input_block_json = j.at("input_block_config");
    p.input_block_config.parallel_gains_config =
        input_block_json.at("parallel_gains_config").get<ParallelGainsOptions>();

    p.input_block_config.single_channel_processors.clear();
    for (const auto& processor_json : input_block_json.at("single_channel_processors"))
    {
        p.input_block_config.single_channel_processors.push_back(SingleChannelProcessorFromJson(processor_json));
    }

    p.input_block_config.multichannel_processors.clear();
    for (const auto& processor_json : input_block_json.at("multichannel_processors"))
    {
        p.input_block_config.multichannel_processors.push_back(MultichannelProcessorFromJson(processor_json));
    }

    p.feedback_matrix_config = FeedbackMatrixFromJson(j.at("feedback_matrix_config"));

    p.loop_filter_configs.clear();
    for (const auto& processor_json : j.at("loop_filter_configs"))
    {
        p.loop_filter_configs.push_back(MultichannelProcessorFromJson(processor_json));
    }

    const auto& output_block_json = j.at("output_block_config");
    p.output_block_config.parallel_gains_config =
        output_block_json.at("parallel_gains_config").get<ParallelGainsOptions>();
    p.output_block_config.single_channel_processors.clear();
    for (const auto& processor_json : output_block_json.at("single_channel_processors"))
    {
        p.output_block_config.single_channel_processors.push_back(SingleChannelProcessorFromJson(processor_json));
    }
    p.output_block_config.multichannel_processors.clear();
    for (const auto& processor_json : output_block_json.at("multichannel_processors"))
    {
        p.output_block_config.multichannel_processors.push_back(MultichannelProcessorFromJson(processor_json));
    }

    p.tone_correction_filters.clear();
    for (const auto& processor_json : j.at("tone_correction_filters"))
    {
        p.tone_correction_filters.push_back(SingleChannelProcessorFromJson(processor_json));
    }
}

} // namespace sfFDN
