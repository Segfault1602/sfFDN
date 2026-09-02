#include "sffdn/fdn_config.h"

#include "json_helper.h"
#include "math_utils.h"

#include "sffdn/dattorro_delay.h"
#include "sffdn/delay.h"
#include "sffdn/delay_time_varying.h"
#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/filter.h"
#include "sffdn/filter_design.h"
#include "sffdn/filter_feedback_matrix.h"
#include "sffdn/filterbank.h"
#include "sffdn/nonlinear.h"
#include "sffdn/parallel_gains.h"
#include "sffdn/schroeder_allpass.h"
#include "sffdn/time_varying_feedback_matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <variant>

namespace
{
template <typename T>
std::string VariantTypeName()
{
    if constexpr (std::is_same_v<T, sfFDN::ParallelGainsOptions>)
    {
        return "ParallelGainsOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelSchroederAllpassSectionOptions>)
    {
        return "MultichannelSchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions>)
    {
        return "MultichannelTimeVaryingSchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelDattorroDelayOptions>)
    {
        return "MultichannelDattorroDelayOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::AttenuationFilterBankOptions>)
    {
        return "AttenuationFilterBankOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::SchroederAllpassSectionOptions>)
    {
        return "SchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::TimeVaryingSchroederAllpassSectionOptions>)
    {
        return "TimeVaryingSchroederAllpassSectionOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::AllpassFilterOptions>)
    {
        return "AllpassFilterOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::CascadedBiquadsOptions>)
    {
        return "CascadedBiquadsOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::FirOptions>)
    {
        return "FirOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::DelayOptions>)
    {
        return "DelayOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::DattorroDelayOptions>)
    {
        return "DattorroDelayOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::DelayBankOptions>)
    {
        return "DelayBankOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::DelayBankTimeVaryingOptions>)
    {
        return "DelayBankTimeVaryingOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixOptions>)
    {
        return "CascadedFeedbackMatrixInfo";
    }
    else if constexpr (std::is_same_v<T, sfFDN::ScalarFeedbackMatrixOptions>)
    {
        return "ScalarFeedbackMatrixOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::ControllableFullWaveRectifierOptions>)
    {
        return "ControllableFullWaveRectifierOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::SignalDependentFractionalDelayOptions>)
    {
        return "SignalDependentFractionalDelayOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::RingModulatorOptions>)
    {
        return "RingModulatorOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelControllableFullWaveRectifierOptions>)
    {
        return "MultichannelControllableFullWaveRectifierOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelSignalDependentFractionalDelayOptions>)
    {
        return "MultichannelSignalDependentFractionalDelayOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelRingModulatorOptions>)
    {
        return "MultichannelRingModulatorOptions";
    }
    else if constexpr (std::is_same_v<T, sfFDN::MultichannelFirOptions>)
    {
        return "MultichannelFirOptions";
    }
    else
    {
        throw std::runtime_error("Unsupported variant type");
    }
}

bool ValidateDelayBank(const sfFDN::DelayBankOptions& option, const sfFDN::FDNConfig& config)
{
    if (option.delays.size() != config.fdn_size)
    {
        std::cerr << "Delay bank config must have the same number of delays as the FDN size\n";
        return false;
    }

    return true;
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
            [&config](const sfFDN::MultichannelSchroederAllpassSectionOptions& schroeder_config) {
                if (schroeder_config.sections.size() != config.fdn_size)
                {
                    std::cerr << "Number of sections in multichannel Schroeder allpass config must match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions& schroeder_config) {
                if (schroeder_config.sections.size() != config.fdn_size)
                {
                    std::cerr << "Number of sections in multichannel time-varying Schroeder allpass config must match "
                                 "FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelDattorroDelayOptions& dattorro_config) {
                if (dattorro_config.delays.size() != config.fdn_size)
                {
                    std::cerr << "Number of delays in multichannel Dattorro delay config must match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelControllableFullWaveRectifierOptions& rectifier_config) {
                if (rectifier_config.channels.size() != config.fdn_size)
                {
                    std::cerr
                        << "Number of channels in multichannel full-wave rectifier config must match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelSignalDependentFractionalDelayOptions& sdfd_config) {
                if (sdfd_config.channels.size() != config.fdn_size)
                {
                    std::cerr << "Number of channels in multichannel signal-dependent fractional delay config must "
                                 "match FDN size\n";
                    return false;
                }
                return true;
            },
            [&config](const sfFDN::MultichannelRingModulatorOptions& ring_mod_config) {
                if (ring_mod_config.channels.size() != config.fdn_size)
                {
                    std::cerr << "Number of channels in multichannel ring modulator config must match FDN size\n";
                    return false;
                }
                return true;
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
            [](const auto&) { return true; }},
        processor_options);
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

    if (!ValidateDelayBank(config.delay_bank_config, config))
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
        !ValidateConfig(config.attenuation_filter_bank_config.value(), config))
    {
        return false;
    }

    if (std::ranges::any_of(config.loop_filter_configs, [&config](const auto& processor_config) {
            return !ValidateConfig(processor_config, config);
        }))
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

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::TimeVaryingSchroederAllpassSectionOptions& config) const
    {
        try
        {
            return std::make_unique<sfFDN::TimeVaryingSchroederAllpassSection>(config);
        }
        catch (const std::invalid_argument& error)
        {
            throw std::runtime_error(std::string("Invalid time-varying Schroeder allpass configuration: ") +
                                     error.what());
        }
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

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::DattorroDelayOptions& config) const
    {
        return std::make_unique<sfFDN::DattorroDelay>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ControllableFullWaveRectifierOptions& config) const
    {
        return std::make_unique<sfFDN::ControllableFullWaveRectifier>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::SignalDependentFractionalDelayOptions& config) const
    {
        return std::make_unique<sfFDN::SignalDependentFractionalDelay>(config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::RingModulatorOptions& config) const
    {
        return std::make_unique<sfFDN::RingModulator>(config);
    }
};

struct MultichannelProcessorVisitor
{
    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::ParallelGainsOptions& gains_config) const
    {
        return MakeParallelGainsFromConfig(gains_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelSchroederAllpassSectionOptions& schroeder_config) const
    {
        return sfFDN::MakeMultichannelSchroederAllpassSection(schroeder_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions& schroeder_config) const
    {
        try
        {
            return sfFDN::MakeMultichannelTimeVaryingSchroederAllpassSection(schroeder_config);
        }
        catch (const std::invalid_argument& error)
        {
            throw std::runtime_error(
                std::string("Invalid multichannel time-varying Schroeder allpass configuration: ") + error.what());
        }
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelDattorroDelayOptions& dattorro_config) const
    {
        return sfFDN::MakeMultichannelDattorroDelay(dattorro_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelControllableFullWaveRectifierOptions& rectifier_config) const
    {
        return sfFDN::MakeMultichannelControllableFullWaveRectifier(rectifier_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelSignalDependentFractionalDelayOptions& sdfd_config) const
    {
        return sfFDN::MakeMultichannelSignalDependentFractionalDelay(sdfd_config);
    }

    std::unique_ptr<sfFDN::AudioProcessor> operator()(
        const sfFDN::MultichannelRingModulatorOptions& ring_mod_config) const
    {
        return sfFDN::MakeMultichannelRingModulator(ring_mod_config);
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

    std::unique_ptr<sfFDN::AudioProcessor> operator()(const sfFDN::MultichannelFirOptions& fir_config) const
    {
        auto bank = std::make_unique<sfFDN::FilterBank>();
        for (const auto& coeffs : fir_config.coeffs)
        {
            auto fir = sfFDN::MakeFirFilter(sfFDN::FirOptions{coeffs});
            bank->AddFilter(std::move(fir));
        }
        return bank;
    }
};

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
    fdn->SetDelayBank(config.delay_bank_config);

    // Input gain Block
    fdn->SetInputGains(CreateInputGainsFromConfig(config));

    // Feedback matrix block
    try
    {
        fdn->SetFeedbackMatrix(std::visit(FeedbackMatrixVisitor{}, config.feedback_matrix_config));
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
            fdn->SetLoopFilter(std::move(processor));
        }
        else if (!config.loop_filter_configs.empty())
        {
            auto loop_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);

            if (attenuation_filter_bank != nullptr)
            {
                loop_filter_chain->AddProcessor(std::move(attenuation_filter_bank));
            }

            for (const auto& processor_config : config.loop_filter_configs)
            {
                auto updated_config = UpdateAttenuationFilterBank(processor_config, config);
                auto processor = std::visit(MultichannelProcessorVisitor{}, updated_config);
                loop_filter_chain->AddProcessor(std::move(processor));
            }
            fdn->SetLoopFilter(std::move(loop_filter_chain));
        }
    }
    else
    {
        fdn->SetLoopFilter(std::move(attenuation_filter_bank));
    }

    // TC filters
    if (!config.tone_correction_filters.empty())
    {
        if (config.tone_correction_filters.size() == 1)
        {
            auto processor = std::visit(SingleChannelProcessorVisitor{}, config.tone_correction_filters[0]);
            fdn->SetTCFilter(std::move(processor));
        }
        else
        {
            auto tc_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);
            for (const auto& processor_config : config.tone_correction_filters)
            {
                auto processor = std::visit(SingleChannelProcessorVisitor{}, processor_config);
                tc_filter_chain->AddProcessor(std::move(processor));
            }
            fdn->SetTCFilter(std::move(tc_filter_chain));
        }
        auto tc_filter_chain = std::make_unique<AudioProcessorChain>(config.block_size);
        for (const auto& processor_config : config.tone_correction_filters)
        {
            auto processor = std::visit(SingleChannelProcessorVisitor{}, processor_config);
            tc_filter_chain->AddProcessor(std::move(processor));
        }
        fdn->SetTCFilter(std::move(tc_filter_chain));
    }

    // Output gain block
    fdn->SetOutputGains(CreateOutputGainsFromConfig(config));

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

    if (!j.at("attenuation_filter_bank_config").is_null())
    {
        p.attenuation_filter_bank_config = j.at("attenuation_filter_bank_config")
                                               .at("AttenuationFilterBankOptions")
                                               .get<AttenuationFilterBankOptions>();
    }
    else
    {
        p.attenuation_filter_bank_config.reset();
    }

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
