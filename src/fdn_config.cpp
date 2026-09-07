#include "sffdn/fdn_config.h"

#include "processor_factory.h"

#include "sffdn/channel_matrix.h"
#include "sffdn/delaybank.h"
#include "sffdn/delaybank_time_varying.h"
#include "sffdn/delay_utils.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/fdn.h"
#include "sffdn/filter_design.h"
#include "sffdn/filter_feedback_matrix.h"
#include "sffdn/filterbank.h"
#include "sffdn/parallel_gains.h"
#include "sffdn/time_varying_feedback_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
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

sfFDN::ParallelGainsOptions MakeStageGainsOptions(const sfFDN::StageGainsOptions& stage_options,
                                                  sfFDN::ParallelGainsMode mode)
{
    return {.mode = mode, .gains = stage_options.gains, .time_varying_config = stage_options.time_varying_config};
}

std::unique_ptr<sfFDN::AudioProcessor> CreateStageRoutingFromConfig(const sfFDN::StageGainsOptions& stage_options,
                                                                    sfFDN::ParallelGainsMode mode)
{
    if (!stage_options.time_varying_config.empty())
    {
        return MakeParallelGainsFromConfig(MakeStageGainsOptions(stage_options, mode));
    }

    const uint32_t gain_count = static_cast<uint32_t>(stage_options.gains.size());
    if (mode == sfFDN::ParallelGainsMode::Split)
    {
        return std::make_unique<sfFDN::ChannelMatrix>(sfFDN::ChannelMatrixOptions{
            .input_channel_count = 1U,
            .output_channel_count = gain_count,
            .coefficients = stage_options.gains,
        });
    }
    if (mode == sfFDN::ParallelGainsMode::Merge)
    {
        return std::make_unique<sfFDN::ChannelMatrix>(sfFDN::ChannelMatrixOptions{
            .input_channel_count = gain_count,
            .output_channel_count = 1U,
            .coefficients = stage_options.gains,
        });
    }

    throw std::logic_error("FDN stage routing must use Split or Merge mode");
}

/** Builds one single-channel chain, or nullptr when nothing is configured. */
std::unique_ptr<sfFDN::AudioProcessor> CreateSingleChannelChain(
    const std::vector<sfFDN::single_channel_processor_variant_t>& configs, uint32_t block_size, const char* context)
{
    if (configs.empty())
    {
        return nullptr;
    }
    if (configs.size() == 1)
    {
        return sfFDN::CreateSingleChannelProcessor(configs[0]);
    }

    auto chain = std::make_unique<sfFDN::AudioProcessorChain>(block_size);
    for (const auto& processor_config : configs)
    {
        AddProcessorOrThrow(*chain, sfFDN::CreateSingleChannelProcessor(processor_config), context);
    }
    return chain;
}

/** Appends a single-channel stage, replicating the whole ordered chain once per channel when it is not mono.
 *
 * Each replica is an independent instance, so per-channel filter state never aliases. A mono stage is appended
 * flat, keeping the one-in, one-out construction identical to a network without external channels.
 */
void AddSingleChannelStage(sfFDN::AudioProcessorChain& chain,
                           const std::vector<sfFDN::single_channel_processor_variant_t>& configs, uint32_t block_size,
                           uint32_t channel_count, const char* context)
{
    if (configs.empty())
    {
        return;
    }

    if (channel_count == 1U)
    {
        for (const auto& processor_config : configs)
        {
            AddProcessorOrThrow(chain, sfFDN::CreateSingleChannelProcessor(processor_config), context);
        }
        return;
    }

    auto bank = std::make_unique<sfFDN::FilterBank>();
    for (uint32_t channel = 0; channel < channel_count; ++channel)
    {
        bank->AddFilter(CreateSingleChannelChain(configs, block_size, context));
    }
    AddProcessorOrThrow(chain, std::move(bank), context);
}

std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const sfFDN::FDNConfig& config)
{
    std::unique_ptr<sfFDN::AudioProcessor> input_gains =
        config.input_block_config.boundary_matrix.has_value()
            ? std::unique_ptr<sfFDN::AudioProcessor>(
                  std::make_unique<sfFDN::ChannelMatrix>(*config.input_block_config.boundary_matrix))
            : CreateStageRoutingFromConfig(config.input_block_config.parallel_gains_config,
                                           sfFDN::ParallelGainsMode::Split);

    if (config.input_block_config.single_channel_processors.empty() &&
        config.input_block_config.multichannel_processors.empty())
    {
        return input_gains;
    }

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(config.block_size);

    AddSingleChannelStage(*chain_processor, config.input_block_config.single_channel_processors, config.block_size,
                          config.input_channel_count, "input single-channel processor");

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
        config.output_block_config.boundary_matrix.has_value()
            ? std::unique_ptr<sfFDN::AudioProcessor>(
                  std::make_unique<sfFDN::ChannelMatrix>(*config.output_block_config.boundary_matrix))
            : CreateStageRoutingFromConfig(config.output_block_config.parallel_gains_config,
                                           sfFDN::ParallelGainsMode::Merge);

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

    AddSingleChannelStage(*chain_processor, config.output_block_config.single_channel_processors, config.block_size,
                          config.output_channel_count, "output single-channel processor");

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
            std::visit(
                sfFDN::overloaded{
                    [&](auto& arg) {
                        if (arg.delay <= 0.f)
                        {
                            arg.delay = config.delay_bank_config.delays[i];
                        }
                    },
                },
                filter_config);
        }
        return updated_config;
    }

    return processor_config;
}

/** Tone correction is a per-output-channel filter, so a multichannel output gets one independent replica per channel.
 */
std::unique_ptr<sfFDN::AudioProcessor> CreateToneCorrectionFromConfig(const sfFDN::FDNConfig& config)
{
    if (config.tone_correction_filters.empty())
    {
        return nullptr;
    }
    if (config.output_channel_count == 1U)
    {
        return CreateSingleChannelChain(config.tone_correction_filters, config.block_size,
                                        "tone correction filter");
    }

    auto bank = std::make_unique<sfFDN::FilterBank>();
    for (uint32_t channel = 0; channel < config.output_channel_count; ++channel)
    {
        bank->AddFilter(
            CreateSingleChannelChain(config.tone_correction_filters, config.block_size, "tone correction filter"));
    }
    return bank;
}

} // namespace

namespace sfFDN
{
FDNConfig MakeDefaultFDNConfig(uint32_t fdn_size, uint32_t block_size, float sample_rate)
{
    if (fdn_size == 0)
    {
        throw std::invalid_argument("FDN size must be greater than zero");
    }
    if (block_size == 0)
    {
        throw std::invalid_argument("block size must be greater than zero");
    }
    if (sample_rate <= 0.f)
    {
        throw std::invalid_argument("sample rate must be positive");
    }

    constexpr float kMinimumDelaySeconds = 0.020f;
    constexpr float kMaximumDelaySeconds = 0.050f;
    constexpr uint32_t kDelaySeed = 0x5F4E3D2CU;
    const float minimum_delay = std::max(static_cast<float>(block_size), std::ceil(sample_rate * kMinimumDelaySeconds));
    const float maximum_delay = std::max(minimum_delay + 1.f, std::ceil(sample_rate * kMaximumDelaySeconds));
    const float normalized_gain = 1.f / std::sqrt(static_cast<float>(fdn_size));

    FDNConfig config;
    config.fdn_size = fdn_size;
    config.input_channel_count = 1U;
    config.output_channel_count = 1U;
    config.transposed = false;
    config.direct_gain = 0.f;
    config.block_size = block_size;
    config.sample_rate = sample_rate;
    config.delay_bank_config = {
        .delays = GetDelayLengths(fdn_size, minimum_delay, maximum_delay, DelayLengthType::Random, kDelaySeed),
        .block_size = block_size,
    };
    config.input_block_config.parallel_gains_config.gains.assign(fdn_size, normalized_gain);
    config.output_block_config.parallel_gains_config.gains.assign(fdn_size, normalized_gain);
    config.feedback_matrix_config = ScalarFeedbackMatrixOptions{
        .source =
            GeneratedMatrixOptions{
                .matrix_size = fdn_size,
                .generator =
                    (fdn_size & (fdn_size - 1U)) == 0U ? ScalarMatrixType::Hadamard : ScalarMatrixType::Householder,
            },
    };
    config.attenuation_filter_bank_config = AttenuationFilterBankOptions{
        .filter_configs = {HomogenousFilterOptions{.t60 = 1.f, .delay = 0.f, .sample_rate = sample_rate}},
    };

    return config;
}

namespace
{

void RandomizeMatrixSeed(ScalarFeedbackMatrixOptions& options, std::mt19937& generator)
{
    std::visit(overloaded{
                   [&](GeneratedMatrixOptions& source) { source.rng_seed = generator(); },
                   [](MatrixData&) {},
               },
               options.source);
}

void RandomizeMatrixSeed(multi_channel_processor_variant_t& options, std::mt19937& generator)
{
    std::visit(overloaded{
                   [&](CascadedFeedbackMatrixOptions& source) { source.rng_seed = generator(); },
                   [&](ScalarFeedbackMatrixOptions& source) { RandomizeMatrixSeed(source, generator); },
                   [](ParallelGainsOptions&) {},
                   [](MultichannelProcessorOptions&) {},
                   [](AttenuationFilterBankOptions&) {},
                   [](DelayBankOptions&) {},
                   [](DelayBankTimeVaryingOptions&) {},
               },
               options);
}

void RandomizeMatrixSeed(feedback_matrix_variant_t& options, std::mt19937& generator)
{
    std::visit(overloaded{
                   [&](CascadedFeedbackMatrixOptions& source) { source.rng_seed = generator(); },
                   [&](ScalarFeedbackMatrixOptions& source) { RandomizeMatrixSeed(source, generator); },
                   [&](TimeVaryingFeedbackMatrixOptions& source) { source.rng_seed = generator(); },
               },
               options);
}

} // namespace

void RandomizeMatrixSeeds(FDNConfig& config)
{
    std::mt19937 generator(std::random_device{}());

    RandomizeMatrixSeed(config.feedback_matrix_config, generator);
    for (auto& options : config.input_block_config.multichannel_processors)
    {
        RandomizeMatrixSeed(options, generator);
    }
    for (auto& options : config.output_block_config.multichannel_processors)
    {
        RandomizeMatrixSeed(options, generator);
    }
    for (auto& options : config.loop_filter_configs)
    {
        RandomizeMatrixSeed(options, generator);
    }
}

std::unique_ptr<FDN> CreateFDNFromConfig(const FDNConfig& config)
{
    auto validation = ValidateFDNConfig(config);
    if (!validation.has_value())
    {
        throw FDNConfigError(std::move(validation.error()));
    }
    auto fdn = std::make_unique<FDN>(FDNTopology{
        .order = config.fdn_size,
        .block_size = config.block_size,
        .input_channel_count = config.input_channel_count,
        .output_channel_count = config.output_channel_count,
        .transposed = config.transposed,
    });

    if (config.direct_matrix.has_value())
    {
        if (!fdn->SetDirectPath(std::make_unique<ChannelMatrix>(*config.direct_matrix)))
        {
            throw std::runtime_error("Failed to set FDN direct path");
        }
    }
    else if (config.input_channel_count == config.output_channel_count)
    {
        // The scalar gain is a diagonal path. Validation already rejects a nonzero gain when the counts differ, so a
        // mismatch here means the gain is zero and the direct contribution is silent anyway.
        fdn->SetDirectGain(config.direct_gain);
    }

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
    if (!fdn->SetTCFilter(CreateToneCorrectionFromConfig(config)))
    {
        throw std::runtime_error("Failed to set FDN tone correction filter");
    }

    // Output gain block
    if (!fdn->SetOutputGains(CreateOutputGainsFromConfig(config)))
    {
        throw std::runtime_error("Failed to set FDN output gains");
    }

    return fdn;
}

} // namespace sfFDN
