#include "sffdn/fdn_config.h"

#include "json_helper.h"
#include "math_utils.h"
#include "processor_factory.h"

#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/filter_design.h"
#include "sffdn/filter_feedback_matrix.h"
#include "sffdn/filterbank.h"
#include "sffdn/parallel_gains.h"
#include "sffdn/time_varying_feedback_matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace
{
bool ValidateDelayBank(const sfFDN::DelayBankOptions& option, const sfFDN::FDNConfig& config)
{
    if (option.delays.size() != config.fdn_size)
    {
        std::cerr << "Delay bank config must have the same number of delays as the FDN size\n";
        return false;
    }

    constexpr uint32_t kMaxCapacity = std::numeric_limits<uint32_t>::max();
    if (option.block_size > kMaxCapacity / 2U)
    {
        std::cerr << "Delay bank block size is too large\n";
        return false;
    }

    const uint32_t block_padding = option.block_size * 2U;
    for (const float delay : option.delays)
    {
        if (!std::isfinite(delay) || delay < 0.f)
        {
            std::cerr << "Delay bank contains an unsupported delay\n";
            return false;
        }

        // Matching float arithmetic but comparing in double keeps the integer limit exact.
        const float buffer_capacity = delay + static_cast<float>(block_padding);
        if (static_cast<double>(buffer_capacity) > static_cast<double>(kMaxCapacity))
        {
            std::cerr << "Delay bank contains an unsupported delay\n";
            return false;
        }
    }

    return true;
}

bool ValidatePrimaryDelayBank(const sfFDN::DelayBankOptions& option, const sfFDN::FDNConfig& config)
{
    if (!ValidateDelayBank(option, config))
    {
        return false;
    }

    if (option.block_size == 0 || option.block_size < config.block_size)
    {
        std::cerr << "Primary delay bank block size must be at least the FDN block size\n";
        return false;
    }

    for (const float delay : option.delays)
    {
        if (delay < static_cast<float>(config.block_size))
        {
            std::cerr << "Primary delay bank contains a delay smaller than the FDN block size\n";
            return false;
        }
    }

    return true;
}

bool ValidateAttenuationFilterBank(const sfFDN::AttenuationFilterBankOptions& option, uint32_t fdn_size,
                                   bool allow_shared_config)
{
    const size_t count = option.filter_configs.size();
    if (count == fdn_size || (allow_shared_config && count == 1U))
    {
        return true;
    }

    std::cerr << "Attenuation filter bank must have " << (allow_shared_config ? "one or FDN-size" : "FDN-size")
              << " filter configurations\n";
    return false;
}

bool ValidateDelayBank(const sfFDN::DelayBankTimeVaryingOptions& option, const sfFDN::FDNConfig& config)
{
    if (option.delays.size() != config.fdn_size)
    {
        std::cerr << "Delay bank config must have the same number of delays as the FDN size\n";
        return false;
    }

    return true;
}

bool ValidateMatrix(const sfFDN::feedback_matrix_variant_t& matrix_options, const sfFDN::FDNConfig& config)
{
    return std::visit(
        sfFDN::overloaded{[&config](const sfFDN::CascadedFeedbackMatrixOptions& options) {
                              if (options.matrix_size != config.fdn_size)
                              {
                                  std::cerr << "Cascaded feedback matrix size must match FDN size\n";
                                  return false;
                              }
                              return true;
                          },
                          [&config](const sfFDN::ScalarFeedbackMatrixOptions& options) {
                              if (options.matrix_size != config.fdn_size)
                              {
                                  std::cerr << "Scalar feedback matrix size must match FDN size\n";
                                  return false;
                              }

                              if (options.custom_matrix.has_value() &&
                                  options.custom_matrix->size() != config.fdn_size * config.fdn_size)
                              {
                                  std::cerr << "Custom feedback matrix size must be equal to FDN size squared\n";
                                  return false;
                              }

                              if (!options.custom_matrix.has_value() &&
                                  options.type == sfFDN::ScalarMatrixType::Hadamard &&
                                  !sfFDN::Math::IsPowerOfTwo(config.fdn_size))
                              {
                                  std::cerr << "Hadamard feedback matrix requires FDN size to be a power of two\n";
                                  return false;
                              }

                              return true;
                          },
                          [&config](const sfFDN::TimeVaryingFeedbackMatrixOptions& options) {
                              if (options.matrix_size != config.fdn_size)
                              {
                                  std::cerr << "Time-varying feedback matrix size must match FDN size\n";
                                  return false;
                              }

                              if (options.matrix_size < 2U || (options.matrix_size % 2U) != 0U ||
                                  (options.mode != sfFDN::TimeVaryingMatrixMode::Hadamard &&
                                   options.mode != sfFDN::TimeVaryingMatrixMode::RealSchur) ||
                                  (options.mode == sfFDN::TimeVaryingMatrixMode::Hadamard &&
                                   !sfFDN::Math::IsPowerOfTwo(options.matrix_size)))
                              {
                                  std::cerr << "Time-varying feedback matrix size must be even and, for Hadamard mode, "
                                               "a power of two\n";
                                  return false;
                              }

                              if (options.mode == sfFDN::TimeVaryingMatrixMode::Hadamard &&
                                  !options.time_varying_config.empty() &&
                                  options.time_varying_config.size() != options.matrix_size / 2U)
                              {
                                  std::cerr << "Hadamard time-varying feedback matrix requires one modulation option "
                                               "per rotation block\n";
                                  return false;
                              }

                              for (const auto& modulation : options.time_varying_config)
                              {
                                  if (!std::isfinite(modulation.frequency) ||
                                      !(std::abs(modulation.amplitude) <= 1.0F) ||
                                      !std::isfinite(modulation.initial_phase) || modulation.initial_phase < 0.0F ||
                                      modulation.initial_phase > 1.0F)
                                  {
                                      std::cerr << "Time-varying feedback matrix modulation parameters are invalid\n";
                                      return false;
                                  }
                              }

                              return true;
                          }},
        matrix_options);
}

bool ValidateConfig(const sfFDN::multi_channel_processor_variant_t& processor_options, const sfFDN::FDNConfig& config)
{
    return std::visit(
        sfFDN::overloaded{
            [&config](const sfFDN::ParallelGainsOptions& gains_config) {
                if (gains_config.mode != sfFDN::ParallelGainsMode::Parallel)
                {
                    std::cerr << "Parallel gains config in multi-channel processor block must be in Parallel mode\n";
                    return false;
                }
                if (gains_config.gains.size() != config.fdn_size)
                {
                    std::cerr << "Number of gains in parallel gains config must match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelProcessorOptions& processor_config) {
                if (processor_config.channels.size() != config.fdn_size)
                {
                    std::cerr << "Number of channels in multichannel processor config must match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::AttenuationFilterBankOptions& attenuation_config) {
                return ValidateAttenuationFilterBank(attenuation_config, config.fdn_size, false);
            },
            [&config](const sfFDN::DelayBankOptions& delay_bank_config) {
                return ValidateDelayBank(delay_bank_config, config);
            },
            [&config](const sfFDN::DelayBankTimeVaryingOptions& delay_bank_config) {
                return ValidateDelayBank(delay_bank_config, config);
            },
            [&config](const sfFDN::CascadedFeedbackMatrixOptions& matrix_config) {
                const sfFDN::feedback_matrix_variant_t matrix_variant = matrix_config;
                return ValidateMatrix(matrix_variant, config);
            },
            [&config](const sfFDN::ScalarFeedbackMatrixOptions& matrix_config) {
                const sfFDN::feedback_matrix_variant_t matrix_variant = matrix_config;
                return ValidateMatrix(matrix_variant, config);
            },
            [](const auto&) { return true; },},
        processor_options);
}

bool ValidateLoopFilterConfig(const sfFDN::multi_channel_processor_variant_t& processor_options,
                              const sfFDN::FDNConfig& config)
{
    if (const auto* attenuation_config = std::get_if<sfFDN::AttenuationFilterBankOptions>(&processor_options))
    {
        return ValidateAttenuationFilterBank(*attenuation_config, config.fdn_size, true);
    }

    return ValidateConfig(processor_options, config);
}

bool ValidateConfig(const sfFDN::FDNConfig& config)
{
    if (config.fdn_size == 0)
    {
        std::cerr << "FDN size must be greater than 0\n";
        return false;
    }

    if (config.block_size == 0)
    {
        std::cerr << "Block size must be greater than 0\n";
        return false;
    }

    if (config.sample_rate <= 0.f)
    {
        std::cerr << "Sample rate must be greater than 0\n";
        return false;
    }

    if (!ValidatePrimaryDelayBank(config.delay_bank_config, config))
    {
        return false;
    }

    const auto& input_gains_config = config.input_block_config.parallel_gains_config;
    if (input_gains_config.mode != sfFDN::ParallelGainsMode::Split ||
        input_gains_config.gains.size() != config.fdn_size)
    {
        std::cerr << "Number of gains in input parallel gains config must match FDN size and be in Split mode\n";
        return false;
    }

    const auto& output_gains_config = config.output_block_config.parallel_gains_config;
    if (output_gains_config.mode != sfFDN::ParallelGainsMode::Merge ||
        output_gains_config.gains.size() != config.fdn_size)
    {
        std::cerr << "Number of gains in output parallel gains config must match FDN size and be in Merge mode\n";
        return false;
    }

    if (!ValidateMatrix(config.feedback_matrix_config, config))
    {
        return false;
    }

    if (std::ranges::any_of(config.input_block_config.multichannel_processors, [&config](const auto& processor_config) {
            return !ValidateConfig(processor_config, config);
        }))
    {
        return false;
    }

    if (std::ranges::any_of(
            config.output_block_config.multichannel_processors,
            [&config](const auto& processor_config) { return !ValidateConfig(processor_config, config); }))
    {
        return false;
    }

    if (config.attenuation_filter_bank_config.has_value() &&
        !ValidateAttenuationFilterBank(*config.attenuation_filter_bank_config, config.fdn_size, true))
    {
        return false;
    }

    if (std::ranges::any_of(config.loop_filter_configs, [&config](const auto& processor_config) {
            return !ValidateLoopFilterConfig(processor_config, config);
        }))
    {
        return false;
    }

    return true;
}

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
    if (!ValidateConfig(config))
    {
        throw std::runtime_error("Invalid FDNConfig");
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
