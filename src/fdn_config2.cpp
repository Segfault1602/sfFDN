#include "sffdn/fdn_config2.h"

#include "sffdn/sffdn.h"

#include <cassert>
#include <cstdint>
#include <iostream>
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
            using T = std::decay_t<decltype(matrix_config)>;
            if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixInfo>)
            {
                if (matrix_config.matrix_size != config.fdn_size)
                {
                    std::cerr << "Feedback matrix channel count must match FDN size" << std::endl;
                    return false;
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<float>>)
            {
                if (matrix_config.size() != config.fdn_size * config.fdn_size)
                {
                    std::cerr << "Scalar feedback matrix size must be N x N where N is FDN size" << std::endl;
                    return false;
                }
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
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::SchroederAllpassSectionConfig& config) const
    {
        return std::make_unique<sfFDN::SchroederAllpassSection>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::AllpassFilterConfig& config) const
    {
        auto filter = std::make_unique<sfFDN::AllpassFilter>();
        filter->SetCoefficients(config.coeff);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedBiquadsConfig& config) const
    {
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(config.coeffs);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::FirConfig& config) const
    {
        auto filter = std::make_unique<sfFDN::Fir>();
        filter->SetCoefficients(config.coeffs);
        return filter;
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayConfig& config) const
    {
        return std::make_unique<sfFDN::DelayTimeVarying>(config);
    }
};

struct MultichannelProcessorVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ParallelGainsConfig& gains_config) const
    {
        return MakeParallelGainsFromConfig(gains_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::ParallelSchroederAllpassSectionConfig& schroeder_config) const
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
        const sfFDN::AttenuationFilterBankConfig& attenuation_config) const
    {
        return sfFDN::CreateAttenuationFilterBank(attenuation_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayBankConfig& delay_bank_config) const
    {
        return std::make_unique<sfFDN::DelayBank>(delay_bank_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DelayBankTimeVaryingConfig& delay_bank_config) const
    {
        return std::make_unique<sfFDN::DelayBankTimeVarying>(delay_bank_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedFeedbackMatrixInfo& matrix_config) const
    {
        return std::make_unique<sfFDN::FilterFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ScalarFeedbackMatrixConfig& matrix_config) const
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
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::CascadedFeedbackMatrixInfo& matrix_config) const
    {
        return std::make_unique<sfFDN::FilterFeedbackMatrix>(matrix_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ScalarFeedbackMatrixConfig& matrix_config) const
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

        sfFDN::ScalarFeedbackMatrixConfig scalar_config;
        scalar_config.matrix_size = matrix_size;
        scalar_config.custom_matrix = matrix_config;
        return std::make_unique<sfFDN::ScalarFeedbackMatrix>(scalar_config);
    }
};

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
                loop_filter_chain->AddProcessor(std::visit(MultichannelProcessorVisitor{}, processor_config));
            }
            fdn->SetFilterBank(std::move(loop_filter_chain));
        }
        else
        {
            fdn->SetFilterBank(std::visit(MultichannelProcessorVisitor{}, config.loop_filter_configs[0]));
        }
    }

    // Output gain block
    fdn->SetOutputGains(CreateOutputGainsFromConfig(config));

    return fdn;
}

} // namespace sfFDN
