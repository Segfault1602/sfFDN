#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "sffdn/config_diagnostics.h"
#include "sffdn/sffdn.h"
#include <sffdn/serialization.h>

namespace
{

sfFDN::FDNConfig MakeValidConfig()
{
    constexpr uint32_t kOrder = 4;
    sfFDN::FDNConfig config{};
    config.fdn_size = kOrder;
    config.block_size = 8;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {8.F, 9.F, 10.F, 11.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = kOrder,
                .generator = sfFDN::ScalarMatrixType::Hadamard,
            },
    };
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kOrder, 1.F),
        .time_varying_config = {},
    };
    return config;
}

sfFDN::AttenuationFilterBankOptions MakeAttenuationBank(size_t count)
{
    sfFDN::AttenuationFilterBankOptions bank;
    for (size_t index = 0; index < count; ++index)
    {
        bank.filter_configs.emplace_back(
            sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 0.F, .sample_rate = 48000.F});
    }
    return bank;
}

bool HasIssue(const std::vector<sfFDN::ConfigIssue>& issues, sfFDN::ConfigErrorCode code, std::string_view path)
{
    return std::ranges::any_of(issues, [&](const auto& issue) { return issue.code == code && issue.path == path; });
}

std::vector<sfFDN::ConfigIssue> RequireIssues(const std::expected<void, std::vector<sfFDN::ConfigIssue>>& validation)
{
    REQUIRE_FALSE(validation.has_value());
    return validation.error();
}

template <typename... Types>
concept AllEqualityComparable = (std::equality_comparable<Types> && ...);

std::vector<float> RenderDefaultFDN(sfFDN::FDN& fdn, const sfFDN::FDNConfig& config)
{
    const float longest_delay = *std::ranges::max_element(config.delay_bank_config.delays);
    const auto sample_count = static_cast<uint32_t>(std::ceil(longest_delay)) + config.block_size * 2U;
    const auto block_count = (sample_count + config.block_size - 1U) / config.block_size;
    std::vector<float> input(block_count * config.block_size, 0.F);
    std::vector<float> output(input.size(), 0.F);
    input[0] = 1.F;

    for (uint32_t block = 0; block < block_count; ++block)
    {
        const auto offset = block * config.block_size;
        const auto input_buffer =
            sfFDN::AudioBuffer(config.block_size, 1U, std::span(input).subspan(offset, config.block_size));
        auto output_buffer =
            sfFDN::AudioBuffer(config.block_size, 1U, std::span(output).subspan(offset, config.block_size));
        std::fill_n(output.begin() + offset, config.block_size, 0.F);
        fdn.Process(input_buffer, output_buffer);
    }

    return output;
}

void NormalizeMatrixSeed(sfFDN::ScalarFeedbackMatrixOptions& options)
{
    auto* generated = std::get_if<sfFDN::GeneratedMatrixOptions>(&options.source);
    if (generated != nullptr)
    {
        generated->rng_seed = sfFDN::kDefaultMatrixSeed;
    }
}

void NormalizeMatrixSeed(sfFDN::CascadedFeedbackMatrixOptions& options)
{
    options.rng_seed = sfFDN::kDefaultMatrixSeed;
}

void NormalizeMatrixSeed(sfFDN::TimeVaryingFeedbackMatrixOptions& options)
{
    options.rng_seed = sfFDN::kDefaultMatrixSeed;
}

void NormalizeMatrixSeeds(sfFDN::FDNConfig& config)
{
    const auto normalize = [](auto& options) {
        using Options = std::remove_cvref_t<decltype(options)>;
        if constexpr (std::same_as<Options, sfFDN::ScalarFeedbackMatrixOptions> ||
                      std::same_as<Options, sfFDN::CascadedFeedbackMatrixOptions> ||
                      std::same_as<Options, sfFDN::TimeVaryingFeedbackMatrixOptions>)
        {
            NormalizeMatrixSeed(options);
        }
    };
    std::visit(normalize, config.feedback_matrix_config);
    for (auto* placement : {&config.input_block_config.multichannel_processors,
                            &config.output_block_config.multichannel_processors, &config.loop_filter_configs})
    {
        for (auto& options : *placement)
        {
            std::visit(normalize, options);
        }
    }
}

} // namespace

TEST_CASE("FDNConfig validates and builds usable configurations", "[fdn]")
{
    constexpr std::array kMatrixTypes = {
        sfFDN::ScalarMatrixType::Identity,          sfFDN::ScalarMatrixType::Random,
        sfFDN::ScalarMatrixType::Householder,       sfFDN::ScalarMatrixType::RandomHouseholder,
        sfFDN::ScalarMatrixType::Hadamard,          sfFDN::ScalarMatrixType::Circulant,
        sfFDN::ScalarMatrixType::Allpass,           sfFDN::ScalarMatrixType::NestedAllpass,
        sfFDN::ScalarMatrixType::VariableDiffusion,
    };

    for (const auto type : kMatrixTypes)
    {
        auto config = MakeValidConfig();
        config.input_block_config.multichannel_processors.emplace_back(sfFDN::DelayBankOptions{
            .delays = {1.F, 2.F, 3.F, 4.F},
            .block_size = 4U,
            .interpolation_type = sfFDN::DelayInterpolationType::None,
        });
        config.attenuation_filter_bank_config = MakeAttenuationBank(1U);
        config.loop_filter_configs.emplace_back(MakeAttenuationBank(1U));
        config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
            .source =
                sfFDN::GeneratedMatrixOptions{
                    .matrix_size = config.fdn_size,
                    .generator = type == sfFDN::ScalarMatrixType::VariableDiffusion
                                     ? sfFDN::MatrixGeneratorOptions{sfFDN::VariableDiffusionOptions{.diffusion = 0.5F}}
                                     : sfFDN::MatrixGeneratorOptions{type},
                    .rng_seed = 42U,
                },
        };

        REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
        REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(config));
    }

    auto variants = MakeValidConfig();
    variants.input_block_config.single_channel_processors = {
        sfFDN::SchroederAllpassSectionOptions{.delays = {2.F}, .gains = {0.25F}},
        sfFDN::TimeVaryingSchroederAllpassSectionOptions{
            .delays = {3.F},
            .gains = {0.25F},
            .time_varying_config = {{.frequency = 0.01F, .amplitude = 0.1F, .initial_phase = 0.F}},
        },
        sfFDN::AllpassFilterOptions{.coeff = 0.25F},
        sfFDN::CascadedBiquadsOptions{.coeffs = {{1.F, 0.F, 0.F, 1.F, 0.F, 0.F}}},
        sfFDN::FirOptions{.coeffs = {1.F}},
        sfFDN::DelayOptions{.delay = 4.F, .max_delay = 8U},
        sfFDN::GraphicEQOptions{
            .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
        },
        sfFDN::DattorroDelayOptions{.delay_config = {.delay = 4.F, .max_delay = 8U}},
        sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.5F},
        sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5F},
        sfFDN::RingModulatorOptions{.frequency = 0.01F},
    };
    variants.input_block_config.multichannel_processors = {
        sfFDN::ParallelGainsOptions{
            .mode = sfFDN::ParallelGainsMode::Parallel, .gains = {1.F, 1.F, 1.F, 1.F}, .time_varying_config = {}},
        sfFDN::MultichannelProcessorOptions{.channels = {std::nullopt, std::nullopt, std::nullopt, std::nullopt}},
        MakeAttenuationBank(4U),
        sfFDN::DelayBankOptions{.delays = {1.F, 2.F, 3.F, 4.F}, .block_size = 4U},
        sfFDN::DelayBankTimeVaryingOptions{.delays = {3.F, 4.F, 5.F, 6.F}, .max_delay = 8U, .time_varying_config = {}},
        sfFDN::CascadedFeedbackMatrixOptions{.matrix_size = 4U, .stage_count = 1U},
        sfFDN::ScalarFeedbackMatrixOptions{
            .source =
                sfFDN::MatrixData{
                    4U,
                    {1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F},
                }},
        sfFDN::KroneckerFeedbackMatrixOptions{
            .matrix_size = 4U, .angles = {}, .kernel_types = {}},
        sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions{
            .matrix = {.matrix_size = 4U, .angles = {}, .kernel_types = {}},
            .time_varying_config = {},
        },
    };
    REQUIRE(sfFDN::ValidateFDNConfig(variants).has_value());

    variants.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{.matrix_size = 4U, .stage_count = 1U};
    REQUIRE(sfFDN::ValidateFDNConfig(variants).has_value());
    variants.feedback_matrix_config = sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = 4U,
        .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
        .time_varying_config = {},
    };
    REQUIRE(sfFDN::ValidateFDNConfig(variants).has_value());
    variants.feedback_matrix_config = sfFDN::KroneckerFeedbackMatrixOptions{
        .matrix_size = 4U, .angles = {}, .kernel_types = {}};
    REQUIRE(sfFDN::ValidateFDNConfig(variants).has_value());
    const auto kronecker_fdn = sfFDN::CreateFDNFromConfig(variants);
    const auto rendered = RenderDefaultFDN(*kronecker_fdn, variants);
    REQUIRE(std::ranges::any_of(rendered, [](float sample) { return sample != 0.0F; }));

    variants.feedback_matrix_config = sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions{
        .matrix = {.matrix_size = 4U, .angles = {}, .kernel_types = {}},
        .time_varying_config = {},
    };
    REQUIRE(sfFDN::ValidateFDNConfig(variants).has_value());
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(variants));
}

TEST_CASE("CreateFDNFromConfig selects static matrices and preserves modulated stage gains", "[fdn]")
{
    SECTION("unmodulated stage gains become ChannelMatrix boundaries")
    {
        const auto config = MakeValidConfig();
        const auto before = config;
        auto fdn = sfFDN::CreateFDNFromConfig(config);

        REQUIRE(dynamic_cast<sfFDN::ChannelMatrix*>(fdn->GetInputGains()) != nullptr);
        REQUIRE(dynamic_cast<sfFDN::ChannelMatrix*>(fdn->GetOutputGains()) != nullptr);
        REQUIRE(config == before);
    }

    SECTION("a nonempty zero-amplitude modulation vector remains time-varying")
    {
        auto config = MakeValidConfig();
        config.input_block_config.parallel_gains_config.time_varying_config =
            std::vector<sfFDN::ModulationOptions>(config.fdn_size);
        config.output_block_config.parallel_gains_config.time_varying_config =
            std::vector<sfFDN::ModulationOptions>(config.fdn_size);
        const auto before = config;
        auto fdn = sfFDN::CreateFDNFromConfig(config);

        REQUIRE(dynamic_cast<sfFDN::TimeVaryingParallelGains*>(fdn->GetInputGains()) != nullptr);
        REQUIRE(dynamic_cast<sfFDN::TimeVaryingParallelGains*>(fdn->GetOutputGains()) != nullptr);
        REQUIRE(config == before);
    }

    SECTION("input and output stages select their implementation independently")
    {
        auto input_modulated = MakeValidConfig();
        input_modulated.input_block_config.parallel_gains_config.time_varying_config =
            std::vector<sfFDN::ModulationOptions>(input_modulated.fdn_size);
        auto input_modulated_fdn = sfFDN::CreateFDNFromConfig(input_modulated);
        REQUIRE(dynamic_cast<sfFDN::TimeVaryingParallelGains*>(input_modulated_fdn->GetInputGains()) != nullptr);
        REQUIRE(dynamic_cast<sfFDN::ChannelMatrix*>(input_modulated_fdn->GetOutputGains()) != nullptr);

        auto output_modulated = MakeValidConfig();
        output_modulated.output_block_config.parallel_gains_config.time_varying_config =
            std::vector<sfFDN::ModulationOptions>(output_modulated.fdn_size);
        auto output_modulated_fdn = sfFDN::CreateFDNFromConfig(output_modulated);
        REQUIRE(dynamic_cast<sfFDN::ChannelMatrix*>(output_modulated_fdn->GetInputGains()) != nullptr);
        REQUIRE(dynamic_cast<sfFDN::TimeVaryingParallelGains*>(output_modulated_fdn->GetOutputGains()) != nullptr);
    }

    SECTION("static matrix boundaries compose with the surrounding processor chains")
    {
        auto config = MakeValidConfig();
        config.input_block_config.single_channel_processors = {sfFDN::FirOptions{.coeffs = {1.F}}};
        config.input_block_config.multichannel_processors = {
            sfFDN::ParallelGainsOptions{
                .mode = sfFDN::ParallelGainsMode::Parallel,
                .gains = std::vector<float>(config.fdn_size, 1.F),
                .time_varying_config = {},
            },
        };
        config.output_block_config.multichannel_processors = {
            sfFDN::ParallelGainsOptions{
                .mode = sfFDN::ParallelGainsMode::Parallel,
                .gains = std::vector<float>(config.fdn_size, 1.F),
                .time_varying_config = {},
            },
        };
        config.output_block_config.single_channel_processors = {sfFDN::FirOptions{.coeffs = {1.F}}};
        config.tone_correction_filters = {sfFDN::FirOptions{.coeffs = {1.F}}};
        const auto before = config;

        REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
        auto fdn = sfFDN::CreateFDNFromConfig(config);
        REQUIRE(dynamic_cast<sfFDN::AudioProcessorChain*>(fdn->GetInputGains()) != nullptr);
        REQUIRE(dynamic_cast<sfFDN::AudioProcessorChain*>(fdn->GetOutputGains()) != nullptr);
        REQUIRE(fdn->GetTCFilter() != nullptr);
        REQUIRE(std::ranges::any_of(RenderDefaultFDN(*fdn, config), [](float sample) { return sample != 0.F; }));
        REQUIRE(config == before);
    }
}

TEST_CASE("ValidateFDNConfig reports issues at resolvable JSON pointers", "[fdn]")
{
    auto config = MakeValidConfig();
    config.block_size = 0U;
    config.delay_bank_config.delays[0] = -1.F;
    config.input_block_config.single_channel_processors.emplace_back(
        sfFDN::DelayOptions{.delay = 4.F, .max_delay = 3U});
    config.input_block_config.multichannel_processors.emplace_back(sfFDN::MultichannelProcessorOptions{
        .channels = {sfFDN::FirOptions{.coeffs = {}}, std::nullopt},
    });
    config.attenuation_filter_bank_config = sfFDN::AttenuationFilterBankOptions{
        .filter_configs = {sfFDN::TwoBandFilterOptions{
            .t60s = {1.F, 0.5F},
            .delay = -1.F,
            .sample_rate = 48000.F,
        }},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = config.fdn_size,
                .generator = sfFDN::VariableDiffusionOptions{.diffusion = 2.F},
            },
    };
    config.tone_correction_filters.emplace_back(sfFDN::RingModulatorOptions{.frequency = -1.F});

    const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE_FALSE(issues.empty());
    const nlohmann::json serialized = config;
    for (const auto& issue : issues)
    {
        INFO(issue.path);
        REQUIRE(serialized.contains(nlohmann::json::json_pointer(issue.path)));
    }

    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/block_size"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/delay_bank_config/delays/0"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/input_block_config/single_channel_processors/0/DelayOptions/max_delay"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                     "/input_block_config/multichannel_processors/0/MultichannelProcessorOptions/channels"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/ScalarFeedbackMatrixOptions/source/GeneratedMatrixOptions/generator/"
                     "VariableDiffusionOptions/diffusion"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue,
                     "/tone_correction_filters/0/RingModulatorOptions/frequency"));
}

TEST_CASE("ValidateFDNConfig aggregates independent root issues without dependent noise", "[fdn]")
{
    auto config = MakeValidConfig();
    config.fdn_size = 0U;
    config.block_size = 0U;
    config.sample_rate = 0.F;

    const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
    REQUIRE(issues.size() == 3U);
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/fdn_size"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/block_size"));
    REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/sample_rate"));
}

TEST_CASE("ValidateFDNConfig is deterministic and CreateFDNFromConfig reports its issues", "[fdn]")
{
    auto config = MakeValidConfig();
    config.delay_bank_config.delays[0] = -1.F;
    config.output_block_config.parallel_gains_config.gains.pop_back();
    const auto original = config;

    const auto first = sfFDN::ValidateFDNConfig(config);
    const auto second = sfFDN::ValidateFDNConfig(config);
    const auto issues = RequireIssues(first);
    REQUIRE(issues == RequireIssues(second));
    REQUIRE(config == original);

    try
    {
        static_cast<void>(sfFDN::CreateFDNFromConfig(config));
        FAIL("CreateFDNFromConfig must reject structural errors");
    }
    catch (const sfFDN::FDNConfigError& error)
    {
        REQUIRE(error.Issues() == issues);
        const std::string_view message(error.what());
        for (const auto& issue : issues)
        {
            REQUIRE(message.find(issue.path) != std::string_view::npos);
            REQUIRE(message.find(issue.message) != std::string_view::npos);
        }
    }
}

TEST_CASE("ValidateFDNConfig reports size and capacity boundaries before construction", "[fdn]")
{
    auto dimensions = MakeValidConfig();
    dimensions.fdn_size = 3U;
    dimensions.delay_bank_config.delays.pop_back();
    dimensions.input_block_config.parallel_gains_config.gains.pop_back();
    dimensions.output_block_config.parallel_gains_config.gains.pop_back();
    dimensions.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = 3U,
                .generator = sfFDN::ScalarMatrixType::Hadamard,
            },
    };
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(dimensions)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/ScalarFeedbackMatrixOptions/source/GeneratedMatrixOptions/matrix_size"));

    auto storage = MakeValidConfig();
    storage.block_size = 1073741824U;
    storage.delay_bank_config = {
        .delays = {1073741824.F, 1073741824.F, 1073741824.F, 1073741824.F},
        .block_size = storage.block_size,
    };
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(storage)), sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/fdn_size"));

    auto cascade = MakeValidConfig();
    cascade.fdn_size = 64U;
    cascade.delay_bank_config.delays.assign(64U, 8.F);
    cascade.input_block_config.parallel_gains_config.gains.assign(64U, 1.F);
    cascade.output_block_config.parallel_gains_config.gains.assign(64U, 1.F);
    cascade.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 64U,
        .stage_count = 6U,
        .sparsity = 1.F,
        .generator = sfFDN::ScalarMatrixType::Random,
    };
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(cascade)), sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/stage_count"));
    std::get<sfFDN::CascadedFeedbackMatrixOptions>(cascade.feedback_matrix_config).stage_count = 5U;
    REQUIRE(sfFDN::ValidateFDNConfig(cascade).has_value());

    auto gain = MakeValidConfig();
    gain.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4U,
        .stage_count = 2U,
        .sparsity = 3.F,
        .generator = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 100.F,
    };
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(gain)), sfFDN::ConfigErrorCode::CapacityOverflow,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/gain_per_samples"));
    auto& gain_options = std::get<sfFDN::CascadedFeedbackMatrixOptions>(gain.feedback_matrix_config);
    gain_options.gain_per_samples = -1.F;
    REQUIRE(sfFDN::ValidateFDNConfig(gain).has_value());
    gain_options.sparsity = 1.1F;
    gain_options.gain_per_samples = -0.5F;
    REQUIRE(HasIssue(RequireIssues(sfFDN::ValidateFDNConfig(gain)), sfFDN::ConfigErrorCode::InvalidValue,
                     "/feedback_matrix_config/CascadedFeedbackMatrixInfo/gain_per_samples"));
}

TEST_CASE("FDNConfig compares every configured value structurally", "[fdn]")
{
    static_assert(
        AllEqualityComparable<
            sfFDN::ScalarFeedbackMatrixOptions, sfFDN::CascadedFeedbackMatrixOptions, sfFDN::ModulationOptions,
            sfFDN::TimeVaryingFeedbackMatrixOptions, sfFDN::KroneckerFeedbackMatrixOptions,
            sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions,
            sfFDN::StageGainsOptions, sfFDN::ParallelGainsOptions,
            sfFDN::DelayOptions, sfFDN::DelayBankOptions, sfFDN::DelayBankTimeVaryingOptions, sfFDN::FilterCoefficients,
            sfFDN::AllpassFilterOptions, sfFDN::SparseFirOptions, sfFDN::CascadedBiquadsOptions, sfFDN::FirOptions,
            sfFDN::SchroederAllpassSectionOptions, sfFDN::TimeVaryingSchroederAllpassSectionOptions,
            sfFDN::DattorroDelayOptions, sfFDN::ControllableFullWaveRectifierOptions,
            sfFDN::SignalDependentFractionalDelayOptions, sfFDN::RingModulatorOptions, sfFDN::HomogenousFilterOptions,
            sfFDN::TwoBandFilterOptions, sfFDN::ThreeBandFilterOptions, sfFDN::TenBandFilterOptions,
            sfFDN::AttenuationFilterBankOptions, sfFDN::GraphicEQOptions, sfFDN::MultichannelProcessorOptions,
            sfFDN::InputStageConfig, sfFDN::OutputStageConfig, sfFDN::FDNConfig>);

    const auto original = MakeValidConfig();
    REQUIRE(original == sfFDN::FDNConfig(original));

    auto scalar = original;
    scalar.direct_gain = 0.25F;
    REQUIRE(scalar != original);

    auto vector = original;
    vector.input_block_config.parallel_gains_config.gains[0] = 0.5F;
    REQUIRE(vector != original);

    auto optional = original;
    optional.input_block_config.single_channel_processors.emplace_back(
        sfFDN::DelayOptions{.delay = 4.F, .max_delay = 8U, .lfo_config = std::nullopt});
    auto engaged = optional;
    std::get<sfFDN::DelayOptions>(engaged.input_block_config.single_channel_processors.back()).lfo_config =
        sfFDN::ModulationOptions{};
    REQUIRE(optional != engaged);

    auto variant = original;
    variant.feedback_matrix_config =
        sfFDN::CascadedFeedbackMatrixOptions{.matrix_size = original.fdn_size, .stage_count = 1U};
    REQUIRE(variant != original);

    auto randomized = original;
    randomized.feedback_matrix_config = sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = randomized.fdn_size,
        .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config = {},
    };
    randomized.input_block_config.multichannel_processors = {
        sfFDN::ScalarFeedbackMatrixOptions{.source =
                                               sfFDN::GeneratedMatrixOptions{
                                                   .matrix_size = randomized.fdn_size,
                                                   .generator = sfFDN::ScalarMatrixType::Random,
                                               }},
        sfFDN::ScalarFeedbackMatrixOptions{
            .source =
                sfFDN::MatrixData{
                    randomized.fdn_size,
                    {1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F},
                }},
        sfFDN::CascadedFeedbackMatrixOptions{
            .matrix_size = randomized.fdn_size,
            .stage_count = 1U,
            .generator = sfFDN::ScalarMatrixType::Random,
        },
    };
    const auto before = randomized;
    sfFDN::RandomizeMatrixSeeds(randomized);
    const auto& original_data = std::get<sfFDN::MatrixData>(
        std::get<sfFDN::ScalarFeedbackMatrixOptions>(before.input_block_config.multichannel_processors[1]).source);
    const auto& randomized_data = std::get<sfFDN::MatrixData>(
        std::get<sfFDN::ScalarFeedbackMatrixOptions>(randomized.input_block_config.multichannel_processors[1]).source);
    REQUIRE(randomized_data == original_data);
    NormalizeMatrixSeeds(randomized);
    REQUIRE(randomized == before);
}

TEST_CASE("MakeDefaultFDNConfig creates deterministic usable wet configurations", "[fdn]")
{
    struct DefaultCase
    {
        uint32_t order;
        uint32_t block_size;
        float sample_rate;
    };
    constexpr std::array cases = {
        DefaultCase{3U, 128U, 48000.F},
        DefaultCase{8U, 1024U, 48000.F},
    };

    REQUIRE(sfFDN::MakeDefaultFDNConfig() == sfFDN::MakeDefaultFDNConfig(8U, 128U, 48000.F));
    REQUIRE_THROWS_AS(sfFDN::MakeDefaultFDNConfig(0U), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::MakeDefaultFDNConfig(1U, 0U), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::MakeDefaultFDNConfig(1U, 1U, -1.F), std::invalid_argument);

    for (const auto [order, block_size, sample_rate] : cases)
    {
        const auto config = sfFDN::MakeDefaultFDNConfig(order, block_size, sample_rate);
        REQUIRE(config == sfFDN::MakeDefaultFDNConfig(order, block_size, sample_rate));
        REQUIRE(config.fdn_size == order);
        REQUIRE(config.block_size == block_size);
        REQUIRE(config.sample_rate == sample_rate);
        REQUIRE(config.delay_bank_config.delays.size() == order);
        REQUIRE(config.input_block_config.parallel_gains_config.gains ==
                std::vector<float>(order, 1.F / std::sqrt(static_cast<float>(order))));
        REQUIRE(config.output_block_config.parallel_gains_config.gains ==
                config.input_block_config.parallel_gains_config.gains);

        const auto& matrix = std::get<sfFDN::ScalarFeedbackMatrixOptions>(config.feedback_matrix_config);
        const auto& source = std::get<sfFDN::GeneratedMatrixOptions>(matrix.source);
        REQUIRE(source.generator == ((order & (order - 1U)) == 0U
                                         ? sfFDN::MatrixGeneratorOptions{sfFDN::ScalarMatrixType::Hadamard}
                                         : sfFDN::MatrixGeneratorOptions{sfFDN::ScalarMatrixType::Householder}));
        REQUIRE(config.attenuation_filter_bank_config.has_value());
        REQUIRE(config.attenuation_filter_bank_config->filter_configs.size() == 1U);

        auto first = sfFDN::CreateFDNFromConfig(config);
        auto second = sfFDN::CreateFDNFromConfig(config);
        const auto first_output = RenderDefaultFDN(*first, config);
        REQUIRE(first_output == RenderDefaultFDN(*second, config));
        REQUIRE(std::ranges::any_of(first_output, [](float sample) { return sample != 0.F; }));
    }
}

namespace
{

sfFDN::FDNConfig MakeMimoConfig(uint32_t input_channels, uint32_t output_channels)
{
    auto config = MakeValidConfig();
    config.input_channel_count = input_channels;
    config.output_channel_count = output_channels;
    config.input_block_config.parallel_gains_config = {};
    config.input_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = input_channels,
        .output_channel_count = config.fdn_size,
        .coefficients = std::vector<float>(static_cast<size_t>(input_channels) * config.fdn_size, 0.5F),
    };
    config.output_block_config.parallel_gains_config = {};
    config.output_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = config.fdn_size,
        .output_channel_count = output_channels,
        .coefficients = std::vector<float>(static_cast<size_t>(output_channels) * config.fdn_size, 0.5F),
    };
    return config;
}

} // namespace

TEST_CASE("ValidateFDNConfig accepts boundary matrices that match the declared channel counts", "[fdn]")
{
    const auto config = MakeMimoConfig(2U, 3U);
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto with_direct = config;
    with_direct.direct_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = std::vector<float>(6U, 0.25F),
    };
    REQUIRE(sfFDN::ValidateFDNConfig(with_direct).has_value());
}

TEST_CASE("ValidateFDNConfig rejects ambiguous and mismatched boundary routing", "[fdn]")
{
    SECTION("stage gains and a boundary matrix cannot both be set")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.input_block_config.parallel_gains_config.gains = std::vector<float>(config.fdn_size, 0.5F);
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(
            HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue, "/input_block_config/parallel_gains_config"));
    }

    SECTION("stage modulation and a boundary matrix cannot both be set")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.output_block_config.parallel_gains_config.time_varying_config = {
            {.frequency = 0.001F, .amplitude = 0.1F, .initial_phase = 0.F},
        };
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(
            HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue, "/output_block_config/parallel_gains_config"));
    }

    SECTION("a non-unit channel count requires a boundary matrix")
    {
        auto config = MakeValidConfig();
        config.input_channel_count = 2U;
        config.output_channel_count = 2U;
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/input_block_config/boundary_matrix"));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/output_block_config/boundary_matrix"));
    }

    SECTION("boundary matrix dimensions must match the declared counts and the FDN size")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.input_block_config.boundary_matrix->input_channel_count = 5U;
        config.input_block_config.boundary_matrix->coefficients.assign(5U * config.fdn_size, 0.5F);
        config.output_block_config.boundary_matrix->input_channel_count = 5U;
        config.output_block_config.boundary_matrix->coefficients.assign(5U * 3U, 0.5F);
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                         "/input_block_config/boundary_matrix/input_channel_count"));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch,
                         "/output_block_config/boundary_matrix/input_channel_count"));
    }

    SECTION("boundary matrix coefficient counts must match its dimensions")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.input_block_config.boundary_matrix->coefficients.pop_back();
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(
            HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/input_block_config/boundary_matrix/coefficients"));
    }

    SECTION("channel counts must be greater than zero")
    {
        auto config = MakeValidConfig();
        config.input_channel_count = 0U;
        config.output_channel_count = 0U;
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/input_channel_count"));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::InvalidValue, "/output_channel_count"));
    }
}

TEST_CASE("ValidateFDNConfig constrains the direct path to one representation", "[fdn]")
{
    SECTION("a direct matrix and a nonzero scalar gain are ambiguous")
    {
        auto config = MakeMimoConfig(2U, 2U);
        config.direct_gain = 0.5F;
        config.direct_matrix = sfFDN::ChannelMatrixOptions{
            .input_channel_count = 2U,
            .output_channel_count = 2U,
            .coefficients = std::vector<float>(4U, 0.25F),
        };
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue, "/direct_gain"));
    }

    SECTION("a scalar gain cannot bridge differing input and output channel counts")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.direct_gain = 0.5F;
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::UnsupportedValue, "/direct_gain"));
    }

    SECTION("a zero scalar gain is accepted when the counts differ")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.direct_gain = 0.F;
        REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
    }

    SECTION("a direct matrix must map the declared input count to the declared output count")
    {
        auto config = MakeMimoConfig(2U, 3U);
        config.direct_matrix = sfFDN::ChannelMatrixOptions{
            .input_channel_count = 3U,
            .output_channel_count = 3U,
            .coefficients = std::vector<float>(9U, 0.25F),
        };
        const auto issues = RequireIssues(sfFDN::ValidateFDNConfig(config));
        REQUIRE(HasIssue(issues, sfFDN::ConfigErrorCode::SizeMismatch, "/direct_matrix/input_channel_count"));
    }
}

TEST_CASE("ValidateFDNConfig accepts stage single-channel processors outside a mono boundary", "[fdn]")
{
    // These are replicated per external channel rather than rejected, so a MIMO boundary is valid.
    auto config = MakeMimoConfig(2U, 3U);
    config.input_block_config.single_channel_processors = {sfFDN::FirOptions{.coeffs = {1.F}}};
    config.output_block_config.single_channel_processors = {sfFDN::FirOptions{.coeffs = {1.F}}};
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn->InputChannelCount() == 2U);
    REQUIRE(fdn->OutputChannelCount() == 3U);
}

TEST_CASE("CreateFDNFromConfig builds a MIMO network with the configured routing", "[fdn]")
{
    auto config = MakeMimoConfig(2U, 2U);
    config.block_size = 1U;
    config.delay_bank_config = {.delays = {1.F, 1.F, 1.F, 1.F}, .block_size = 1U};
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source = sfFDN::GeneratedMatrixOptions{.matrix_size = config.fdn_size,
                                                .generator = sfFDN::ScalarMatrixType::Identity},
    };
    // B routes input 0 to delay 0 and input 1 to delay 1; C reads delay 0 into output 0 and delay 1 into output 1.
    config.input_block_config.boundary_matrix->coefficients = {1.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F};
    config.output_block_config.boundary_matrix->coefficients = {1.F, 0.F, 0.F, 0.F, 0.F, 2.F, 0.F, 0.F};
    config.direct_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = 2U,
        .output_channel_count = 2U,
        .coefficients = {0.5F, 0.25F, -0.5F, -0.25F},
    };

    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
    auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn->InputChannelCount() == 2U);
    REQUIRE(fdn->OutputChannelCount() == 2U);
    REQUIRE(fdn->GetDirectPath() != nullptr);

    std::array<float, 2> input = {2.F, 4.F};
    std::array<float, 2> silence{};
    std::array<float, 2> output{};
    const sfFDN::AudioBuffer input_buffer(1U, 2U, input);
    const sfFDN::AudioBuffer silence_buffer(1U, 2U, silence);
    sfFDN::AudioBuffer output_buffer(1U, 2U, output);

    // First block is the direct path alone. D is deliberately off-diagonal so a transposed reading of the row-major
    // coefficients would give [-1, -0.5] instead: D * [2, 4] = [0.5*2 + 0.25*4, -0.5*2 - 0.25*4] = [2, -2].
    fdn->Process(input_buffer, output_buffer);
    REQUIRE(output[0] == 2.F);
    REQUIRE(output[1] == -2.F);

    // Next block returns the unit-delayed wet path: C * [2, 4, 0, 0] = [2, 8].
    output.fill(0.F);
    fdn->Process(silence_buffer, output_buffer);
    REQUIRE(output[0] == 2.F);
    REQUIRE(output[1] == 8.F);
}

TEST_CASE("CreateFDNFromConfig replicates tone correction across output channels", "[fdn]")
{
    auto config = MakeMimoConfig(1U, 2U);
    config.input_channel_count = 1U;
    config.input_block_config.boundary_matrix.reset();
    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(config.fdn_size, 1.F),
        .time_varying_config = {},
    };
    config.tone_correction_filters = {sfFDN::FirOptions{.coeffs = {0.5F}}};
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn->GetTCFilter() != nullptr);
    REQUIRE(fdn->GetTCFilter()->InputChannelCount() == 2U);
    REQUIRE(fdn->GetTCFilter()->OutputChannelCount() == 2U);

    // A mono output keeps the single-channel processor rather than wrapping it in a bank. FilterBank reports its
    // filter count as its channel count, so a one-element bank would also report 1 and the type has to be checked.
    auto mono = MakeValidConfig();
    mono.tone_correction_filters = {sfFDN::FirOptions{.coeffs = {0.5F}}};
    auto mono_fdn = sfFDN::CreateFDNFromConfig(mono);
    REQUIRE(mono_fdn->GetTCFilter()->InputChannelCount() == 1U);
    REQUIRE(dynamic_cast<sfFDN::FilterBank*>(mono_fdn->GetTCFilter()) == nullptr);
    REQUIRE(dynamic_cast<sfFDN::FilterBank*>(fdn->GetTCFilter()) != nullptr);
}

TEST_CASE("CreateFDNFromConfig omits the direct path when the channel counts differ", "[fdn]")
{
    auto config = MakeMimoConfig(1U, 2U);
    config.input_block_config.boundary_matrix.reset();
    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(config.fdn_size, 1.F),
        .time_varying_config = {},
    };
    config.block_size = 1U;
    config.delay_bank_config = {.delays = {1.F, 1.F, 1.F, 1.F}, .block_size = 1U};
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source = sfFDN::GeneratedMatrixOptions{.matrix_size = config.fdn_size,
                                                .generator = sfFDN::ScalarMatrixType::Identity},
    };
    config.output_block_config.boundary_matrix->coefficients.assign(2U * config.fdn_size, 0.F);
    REQUIRE(config.direct_gain == 0.F);
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    // The scalar gain is diagonal and cannot express a 1-to-2 direct path, so the factory installs no direct
    // processor and the dry contribution is silent. With a zeroed output matrix the whole network is silent.
    auto fdn = sfFDN::CreateFDNFromConfig(config);
    REQUIRE(fdn->GetDirectPath() == nullptr);

    std::array<float, 1> input{1.F};
    std::array<float, 2> output{};
    const sfFDN::AudioBuffer input_buffer(1U, 1U, input);
    sfFDN::AudioBuffer output_buffer(1U, 2U, output);
    fdn->Process(input_buffer, output_buffer);
    REQUIRE(output[0] == 0.F);
    REQUIRE(output[1] == 0.F);
}

/** Builds a 2-in, 2-out network whose two external channels are routed through disjoint delay lines.
 *
 * Input channel c feeds delay line c only, and output channel c reads delay line c only, so anything that appears on
 * the wrong output channel is cross-talk rather than mixing.
 */
sfFDN::FDNConfig MakeSeparatedStereoConfig()
{
    auto config = MakeMimoConfig(2U, 2U);
    config.block_size = 1U;
    config.delay_bank_config = {.delays = {1.F, 1.F, 1.F, 1.F}, .block_size = 1U};
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source = sfFDN::GeneratedMatrixOptions{.matrix_size = config.fdn_size,
                                                .generator = sfFDN::ScalarMatrixType::Identity},
    };
    config.input_block_config.boundary_matrix->coefficients = {1.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F};
    config.output_block_config.boundary_matrix->coefficients = {1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F};
    return config;
}

TEST_CASE("CreateFDNFromConfig gives each replicated stage chain independent state", "[fdn]")
{
    auto config = MakeSeparatedStereoConfig();
    // A one-sample delay carries state across blocks, so a single shared instance would leak channel 0 into channel 1.
    const std::vector<sfFDN::single_channel_processor_variant_t> chain = {
        sfFDN::DelayOptions{.delay = 1.F, .max_delay = 4U}};

    uint32_t expected_onset = 0U;
    SECTION("input stage")
    {
        config.input_block_config.single_channel_processors = chain;
        // The stage delay holds the impulse for one block and the delay line for another.
        expected_onset = 2U;
    }
    SECTION("output stage")
    {
        config.output_block_config.single_channel_processors = chain;
        expected_onset = 2U;
    }
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());

    auto fdn = sfFDN::CreateFDNFromConfig(config);
    std::array<float, 2> impulse = {1.F, 0.F};
    std::array<float, 2> silence{};
    std::array<float, 2> output{};
    const sfFDN::AudioBuffer impulse_buffer(1U, 2U, impulse);
    const sfFDN::AudioBuffer silence_buffer(1U, 2U, silence);
    sfFDN::AudioBuffer output_buffer(1U, 2U, output);

    std::vector<float> right_channel;
    std::vector<float> left_channel;
    for (uint32_t block = 0; block < 6U; ++block)
    {
        output.fill(0.F);
        fdn->Process(block == 0 ? impulse_buffer : silence_buffer, output_buffer);
        left_channel.push_back(output[0]);
        right_channel.push_back(output[1]);
    }

    REQUIRE(left_channel[expected_onset - 1U] == 0.F);
    REQUIRE(left_channel[expected_onset] == 1.F);
    // Nothing was ever presented to channel 1, so a shared replica or shared delay line would show up here.
    REQUIRE(std::ranges::all_of(right_channel, [](float sample) { return sample == 0.F; }));
}

TEST_CASE("CreateFDNFromConfig replicates stage single-channel chains faithfully and in order", "[fdn]")
{
    // A ring modulator is time-varying, so it does not commute with a delay: reordering the chain changes the
    // response. Rendering the same ordered chain through a mono network and through both channels of a stereo network
    // pins that each replica is faithful and that its order is preserved.
    const std::vector<sfFDN::single_channel_processor_variant_t> chain = {
        sfFDN::DelayOptions{.delay = 2.F, .max_delay = 8U},
        sfFDN::RingModulatorOptions{.frequency = 0.05F, .amplitude = 1.F, .initial_phase = 0.125F},
    };

    auto stereo = MakeSeparatedStereoConfig();

    // The mono reference uses the same routing as one channel of the stereo network: one delay line, unit gains.
    auto mono = MakeSeparatedStereoConfig();
    mono.input_channel_count = 1U;
    mono.output_channel_count = 1U;
    mono.input_block_config.boundary_matrix.reset();
    mono.output_block_config.boundary_matrix.reset();
    mono.input_block_config.parallel_gains_config = {.gains = {1.F, 0.F, 0.F, 0.F}, .time_varying_config = {}};
    mono.output_block_config.parallel_gains_config = {.gains = {1.F, 0.F, 0.F, 0.F}, .time_varying_config = {}};

    SECTION("input stage")
    {
        stereo.input_block_config.single_channel_processors = chain;
        mono.input_block_config.single_channel_processors = chain;
    }
    SECTION("output stage")
    {
        stereo.output_block_config.single_channel_processors = chain;
        mono.output_block_config.single_channel_processors = chain;
    }
    SECTION("tone correction")
    {
        stereo.tone_correction_filters = chain;
        mono.tone_correction_filters = chain;
    }
    REQUIRE(sfFDN::ValidateFDNConfig(stereo).has_value());
    REQUIRE(sfFDN::ValidateFDNConfig(mono).has_value());

    auto stereo_fdn = sfFDN::CreateFDNFromConfig(stereo);
    auto mono_fdn = sfFDN::CreateFDNFromConfig(mono);

    constexpr uint32_t kBlockCount = 24U;
    std::vector<float> mono_output;
    std::vector<float> stereo_left;
    std::vector<float> stereo_right;
    for (uint32_t block = 0; block < kBlockCount; ++block)
    {
        const float sample = block == 0 ? 1.F : 0.F;

        std::array<float, 1> mono_input{sample};
        std::array<float, 1> mono_block{};
        const sfFDN::AudioBuffer mono_input_buffer(1U, 1U, mono_input);
        sfFDN::AudioBuffer mono_output_buffer(1U, 1U, mono_block);
        mono_fdn->Process(mono_input_buffer, mono_output_buffer);
        mono_output.push_back(mono_block[0]);

        // Drive both stereo channels so the second replica is exercised rather than left idle.
        std::array<float, 2> stereo_input{sample, sample};
        std::array<float, 2> stereo_block{};
        const sfFDN::AudioBuffer stereo_input_buffer(1U, 2U, stereo_input);
        sfFDN::AudioBuffer stereo_output_buffer(1U, 2U, stereo_block);
        stereo_fdn->Process(stereo_input_buffer, stereo_output_buffer);
        stereo_left.push_back(stereo_block[0]);
        stereo_right.push_back(stereo_block[1]);
    }

    REQUIRE(std::ranges::any_of(mono_output, [](float sample) { return sample != 0.F; }));
    REQUIRE(stereo_left == mono_output);
    REQUIRE(stereo_right == mono_output);
}

TEST_CASE("RandomizeMatrixSeeds leaves explicit boundary coefficients unchanged", "[fdn]")
{
    auto config = MakeMimoConfig(2U, 3U);
    config.direct_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = std::vector<float>(6U, 0.25F),
    };
    const auto before = config;
    sfFDN::RandomizeMatrixSeeds(config);
    REQUIRE(config.input_block_config.boundary_matrix == before.input_block_config.boundary_matrix);
    REQUIRE(config.output_block_config.boundary_matrix == before.output_block_config.boundary_matrix);
    REQUIRE(config.direct_matrix == before.direct_matrix);
    REQUIRE(config.input_channel_count == before.input_channel_count);
    REQUIRE(config.output_channel_count == before.output_channel_count);
}
