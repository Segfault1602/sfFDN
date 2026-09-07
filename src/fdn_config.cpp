#include "sffdn/fdn_config.h"

#include "processor_factory.h"

#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/fdn.h"
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
    auto validation = ValidateFDNConfig(config);
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

} // namespace sfFDN
