#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "sffdn/sffdn.h"
#include <sffdn/serialization.h>

namespace
{

sfFDN::FDNConfig MakeRenderableConfig()
{
    sfFDN::FDNConfig config;
    config.fdn_size = 4U;
    config.transposed = false;
    config.direct_gain = 0.F;
    config.block_size = 16U;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {32.F, 37.F, 43.F, 47.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(config.fdn_size, 0.5F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = config.fdn_size,
        .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config =
            {
                {.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.125F},
                {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F},
            },
        .rng_seed = 123U,
    };
    config.loop_filter_configs.emplace_back(sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = config.fdn_size,
                .generator = sfFDN::ScalarMatrixType::Random,
                .rng_seed = 456U,
            },
    });
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(config.fdn_size, 0.5F),
        .time_varying_config = {},
    };
    return config;
}

sfFDN::FDNConfig MakeEveryOptionConfig()
{
    auto config = MakeRenderableConfig();
    config.direct_gain = 0.25F;
    config.input_block_config.parallel_gains_config.time_varying_config = {
        {.frequency = 0.001F, .amplitude = 0.1F, .initial_phase = 0.25F},
        {.frequency = 0.002F, .amplitude = -0.1F, .initial_phase = 0.5F},
        {.frequency = 0.003F, .amplitude = 0.2F, .initial_phase = 0.75F},
        {.frequency = 0.004F, .amplitude = -0.2F, .initial_phase = 1.F},
    };
    config.input_block_config.single_channel_processors = {
        sfFDN::SchroederAllpassSectionOptions{.delays = {2.F}, .gains = {0.25F}},
        sfFDN::TimeVaryingSchroederAllpassSectionOptions{
            .delays = {3.F},
            .gains = {0.25F},
            .time_varying_config = {{.frequency = 0.01F, .amplitude = 0.1F, .initial_phase = 0.F}},
        },
        sfFDN::AllpassFilterOptions{.coeff = 0.25F},
        sfFDN::CascadedBiquadsOptions{.coeffs = {{1.F, 0.F, 0.F, 1.F, 0.F, 0.F}}},
        sfFDN::FirOptions{.coeffs = {1.F, -0.25F}},
        sfFDN::DelayOptions{
            .delay = 3.F,
            .max_delay = 8U,
            .interp_type = sfFDN::DelayInterpolationType::Linear,
            .lfo_config = sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = 0.25F, .initial_phase = 0.F},
        },
        sfFDN::GraphicEQOptions{
            .gains_db = {},
            .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
            .sample_rate = 48000.F,
        },
        sfFDN::DattorroDelayOptions{
            .delay_config = {.delay = 4.F, .max_delay = 8U, .lfo_config = std::nullopt},
            .blend = 0.5F,
            .feedforward = 1.F,
            .feedback = 0.25F,
        },
        sfFDN::ControllableFullWaveRectifierOptions{
            .alpha = 0.5F,
            .antialiasing = true,
            .dc_block = true,
            .sample_rate = 48000.F,
        },
        sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5F},
        sfFDN::RingModulatorOptions{.frequency = 0.01F, .amplitude = 1.F, .initial_phase = 0.25F},
    };

    config.input_block_config.multichannel_processors = {
        sfFDN::ParallelGainsOptions{
            .mode = sfFDN::ParallelGainsMode::Parallel,
            .gains = {1.F, 0.5F, -0.5F, -1.F},
            .time_varying_config = {},
        },
        sfFDN::MultichannelProcessorOptions{
            .channels = {sfFDN::FirOptions{.coeffs = {1.F}}, std::nullopt, sfFDN::AllpassFilterOptions{.coeff = -0.25F},
                         sfFDN::DelayOptions{.delay = 2.F, .max_delay = 4U}},
        },
        sfFDN::AttenuationFilterBankOptions{
            .filter_configs =
                {
                    sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 0.F, .sample_rate = 48000.F},
                    sfFDN::TwoBandFilterOptions{.t60s = {1.F, 0.5F}, .delay = 4.F, .sample_rate = 48000.F},
                    sfFDN::ThreeBandFilterOptions{
                        .t60s = {1.F, 0.75F, 0.5F},
                        .delay = 4.F,
                        .freqs = {800.F, 8000.F},
                        .q = 1.F,
                        .sample_rate = 48000.F,
                    },
                    sfFDN::TenBandFilterOptions{
                        .t60s = {1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F},
                        .delay = 4.F,
                        .sample_rate = 48000.F,
                        .shelf_cutoff = 8000.F,
                    },
                },
        },
        sfFDN::DelayBankOptions{
            .delays = {3.F, 5.F, 7.F, 9.F},
            .block_size = 16U,
            .interpolation_type = sfFDN::DelayInterpolationType::Linear,
        },
        sfFDN::DelayBankTimeVaryingOptions{
            .delays = {3.F, 5.F, 7.F, 9.F},
            .max_delay = 12U,
            .interpolation_type = sfFDN::DelayInterpolationType::Linear,
            .time_varying_config =
                {
                    {.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.F},
                    {.frequency = 0.002F, .amplitude = -0.25F, .initial_phase = 0.25F},
                    {.frequency = 0.003F, .amplitude = 0.5F, .initial_phase = 0.5F},
                    {.frequency = 0.004F, .amplitude = -0.5F, .initial_phase = 0.75F},
                },
        },
        sfFDN::CascadedFeedbackMatrixOptions{
            .matrix_size = 4U,
            .stage_count = 2U,
            .sparsity = 3.F,
            .generator = sfFDN::ScalarMatrixType::Hadamard,
            .gain_per_samples = 0.8F,
            .rng_seed = 17U,
        },
        sfFDN::ScalarFeedbackMatrixOptions{
            .source =
                sfFDN::MatrixData{4U, {1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F}},
        },
        sfFDN::KroneckerFeedbackMatrixOptions{
            .matrix_size = 4U,
            .angles = {0.25F, -0.5F},
            .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection},
        },
        sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions{
            .matrix =
                {
                    .matrix_size = 4U,
                    .angles = {0.125F, -0.25F},
                    .kernel_types = {
                        sfFDN::KroneckerKernelType::Reflection,
                        sfFDN::KroneckerKernelType::Rotation,
                    },
                },
            .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F},
                                    {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F}},
        },
    };
    config.output_block_config.multichannel_processors = {
        sfFDN::ParallelGainsOptions{
            .mode = sfFDN::ParallelGainsMode::Parallel,
            .gains = {1.F, 1.F, 1.F, 1.F},
            .time_varying_config = {},
        },
    };
    config.output_block_config.single_channel_processors = {
        sfFDN::FirOptions{.coeffs = {0.75F}},
    };
    config.attenuation_filter_bank_config =
        std::get<sfFDN::AttenuationFilterBankOptions>(config.input_block_config.multichannel_processors[2]);
    config.tone_correction_filters = {
        sfFDN::FirOptions{.coeffs = {0.5F}},
        sfFDN::AllpassFilterOptions{.coeff = -0.4F},
    };
    return config;
}

sfFDN::FDNConfig MakeMimoConfig()
{
    auto config = MakeRenderableConfig();
    config.input_channel_count = 2U;
    config.output_channel_count = 3U;
    config.direct_gain = 0.F;
    config.direct_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = {0.1F, 0.2F, 0.3F, -0.1F, -0.2F, -0.3F},
    };
    config.input_block_config.parallel_gains_config = {};
    config.input_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = 2U,
        .output_channel_count = config.fdn_size,
        .coefficients = {1.F, 0.F, 0.F, 1.F, 0.5F, 0.5F, 0.5F, -0.5F},
    };
    config.output_block_config.parallel_gains_config = {};
    config.output_block_config.boundary_matrix = sfFDN::ChannelMatrixOptions{
        .input_channel_count = config.fdn_size,
        .output_channel_count = 3U,
        .coefficients = {1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.5F, 0.5F},
    };
    return config;
}

std::vector<float> RenderFDN(sfFDN::FDN& fdn)
{
    constexpr uint32_t kBlockSize = 16U;
    constexpr uint32_t kBlockCount = 16U;
    std::vector<float> input(kBlockSize * kBlockCount, 0.F);
    std::vector<float> output(input.size(), 0.F);
    input[0] = 1.F;

    for (uint32_t block = 0; block < kBlockCount; ++block)
    {
        const auto offset = block * kBlockSize;
        const sfFDN::AudioBuffer input_buffer(kBlockSize, 1U, std::span(input).subspan(offset, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1U, std::span(output).subspan(offset, kBlockSize));
        std::fill(output.begin() + offset, output.begin() + offset + kBlockSize, 0.F);
        fdn.Process(input_buffer, output_buffer);
    }
    return output;
}

std::vector<float> RenderMimoFDN(sfFDN::FDN& fdn)
{
    constexpr uint32_t kBlockSize = 16U;
    constexpr uint32_t kBlockCount = 16U;
    const uint32_t input_channels = fdn.InputChannelCount();
    const uint32_t output_channels = fdn.OutputChannelCount();

    std::vector<float> input(static_cast<size_t>(kBlockSize) * kBlockCount * input_channels, 0.F);
    std::vector<float> output(static_cast<size_t>(kBlockSize) * kBlockCount * output_channels, 0.F);
    const sfFDN::AudioBuffer input_buffer(kBlockSize * kBlockCount, input_channels, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize * kBlockCount, output_channels, output);
    for (uint32_t channel = 0; channel < input_channels; ++channel)
    {
        input[static_cast<size_t>(channel) * kBlockSize * kBlockCount] = 1.F;
    }
    fdn.Process(input_buffer, output_buffer);
    return output;
}

template <typename Options>
void RequireUnchangedAfterFailedRead(const nlohmann::json& malformed, Options options)
{
    const auto before = options;
    REQUIRE_THROWS(malformed.get_to(options));
    REQUIRE(options == before);
}

} // namespace

TEST_CASE("FDNConfig round-trips every configured option through JSON", "[serialization]")
{
    const auto populated = MakeEveryOptionConfig();
    auto absent_optionals = MakeRenderableConfig();
    absent_optionals.attenuation_filter_bank_config.reset();

    auto scalar_feedback = MakeRenderableConfig();
    scalar_feedback.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = 4U,
                .generator = sfFDN::VariableDiffusionOptions{.diffusion = 0.5F},
                .rng_seed = 17U,
            },
    };
    auto cascaded_feedback = MakeRenderableConfig();
    cascaded_feedback.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = 4U,
        .stage_count = 2U,
        .sparsity = 3.F,
        .generator = sfFDN::ScalarMatrixType::Hadamard,
        .gain_per_samples = 0.8F,
        .rng_seed = 17U,
    };
    auto kronecker_feedback = MakeRenderableConfig();
    kronecker_feedback.feedback_matrix_config = sfFDN::KroneckerFeedbackMatrixOptions{
        .matrix_size = 4U,
        .angles = {0.25F, -0.5F},
        .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection},
    };
    auto time_varying_kronecker_feedback = MakeRenderableConfig();
    time_varying_kronecker_feedback.feedback_matrix_config = sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions{
        .matrix =
            {
                .matrix_size = 4U,
                .angles = {0.25F, -0.5F},
                .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection},
            },
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F},
                                {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F}},
    };

    for (const auto& original :
         {populated, absent_optionals, scalar_feedback, cascaded_feedback, kronecker_feedback,
          time_varying_kronecker_feedback, MakeMimoConfig()})
    {
        const nlohmann::json serialized = original;
        const auto round_tripped = serialized.get<sfFDN::FDNConfig>();
        REQUIRE(round_tripped == original);
        REQUIRE(nlohmann::json(round_tripped) == serialized);
    }
}

TEST_CASE("FDNConfig JSON accepts files written before MIMO support", "[serialization]")
{
    const auto legacy_config = MakeRenderableConfig();
    nlohmann::json legacy = legacy_config;

    // Reproduce the pre-MIMO key set exactly.
    legacy.erase("input_channel_count");
    legacy.erase("output_channel_count");
    legacy.erase("direct_matrix");
    legacy["input_block_config"].erase("boundary_matrix");
    legacy["output_block_config"].erase("boundary_matrix");
    REQUIRE(legacy["input_block_config"].size() == 3);
    REQUIRE(legacy["output_block_config"].size() == 3);

    const auto loaded = legacy.get<sfFDN::FDNConfig>();
    REQUIRE(loaded == legacy_config);
    REQUIRE(loaded.input_channel_count == 1U);
    REQUIRE(loaded.output_channel_count == 1U);
    REQUIRE_FALSE(loaded.direct_matrix.has_value());
    REQUIRE_FALSE(loaded.input_block_config.boundary_matrix.has_value());
    REQUIRE_FALSE(loaded.output_block_config.boundary_matrix.has_value());

    // An explicit null is equivalent to an omitted key.
    nlohmann::json nulled = legacy;
    nulled["direct_matrix"] = nullptr;
    nulled["input_block_config"]["boundary_matrix"] = nullptr;
    nulled["output_block_config"]["boundary_matrix"] = nullptr;
    REQUIRE(nulled.get<sfFDN::FDNConfig>() == legacy_config);
}

TEST_CASE("FDNConfig JSON round-trip preserves MIMO rendered output", "[serialization]")
{
    const auto config = MakeMimoConfig();
    REQUIRE(sfFDN::ValidateFDNConfig(config).has_value());
    const auto round_tripped = nlohmann::json(config).get<sfFDN::FDNConfig>();
    REQUIRE(round_tripped == config);

    const auto original_fdn = sfFDN::CreateFDNFromConfig(config);
    const auto round_tripped_fdn = sfFDN::CreateFDNFromConfig(round_tripped);
    REQUIRE(original_fdn->InputChannelCount() == 2U);
    REQUIRE(original_fdn->OutputChannelCount() == 3U);

    const auto rendered = RenderMimoFDN(*original_fdn);
    // Guard against a vacuous comparison of two silent renders.
    REQUIRE(std::ranges::any_of(rendered, [](float sample) { return sample != 0.F; }));
    REQUIRE(rendered == RenderMimoFDN(*round_tripped_fdn));
}

TEST_CASE("FDNConfig JSON round-trip preserves rendered output", "[serialization]")
{
    const auto config = MakeRenderableConfig();
    const auto round_tripped = nlohmann::json(config).get<sfFDN::FDNConfig>();
    REQUIRE(round_tripped == config);

    const auto original_fdn = sfFDN::CreateFDNFromConfig(config);
    const auto round_tripped_fdn = sfFDN::CreateFDNFromConfig(round_tripped);
    REQUIRE(RenderFDN(*original_fdn) == RenderFDN(*round_tripped_fdn));
}

TEST_CASE("Serialization uses canonical JSON contracts", "[serialization]")
{
    const sfFDN::StageGainsOptions stage_gains = {
        .gains = {0.25F, 0.5F},
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F}},
    };
    REQUIRE(nlohmann::json(stage_gains) ==
            nlohmann::json{
                {"gains", {0.25F, 0.5F}},
                {"time_varying_config", {{{"frequency", 0.001F}, {"amplitude", 0.25F}, {"initial_phase", 0.5F}}}},
            });

    const sfFDN::ParallelGainsOptions parallel = {
        .mode = sfFDN::ParallelGainsMode::Parallel,
        .gains = {0.25F, -0.5F},
        .time_varying_config = {},
    };
    REQUIRE(nlohmann::json(parallel) == nlohmann::json{
                                            {"mode", "Parallel"},
                                            {"gains", {0.25F, -0.5F}},
                                            {"time_varying_config", nlohmann::json::array()},
                                        });

    const sfFDN::ScalarFeedbackMatrixOptions generated = {
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = 4U,
                .generator = sfFDN::VariableDiffusionOptions{.diffusion = 0.5F},
                .rng_seed = 17U,
            },
    };
    const nlohmann::json expected_generated = {
        {"source",
         {{"GeneratedMatrixOptions",
           {{"matrix_size", 4U},
            {"generator", {{"VariableDiffusionOptions", {{"diffusion", 0.5F}}}}},
            {"rng_seed", 17U}}}}},
    };
    REQUIRE(nlohmann::json(generated) == expected_generated);

    const sfFDN::ScalarFeedbackMatrixOptions explicit_data = {
        .source = sfFDN::MatrixData{2U, {1.F, 2.F, 3.F, 4.F}},
    };
    const nlohmann::json expected_explicit = {
        {"source", {{"MatrixData", {{"order", 2U}, {"coefficients", {1.F, 2.F, 3.F, 4.F}}}}}},
    };
    REQUIRE(nlohmann::json(explicit_data) == expected_explicit);

    const sfFDN::KroneckerFeedbackMatrixOptions kronecker = {
        .matrix_size = 4U,
        .angles = {0.25F, -0.5F},
        .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection},
    };
    REQUIRE(nlohmann::json(kronecker) ==
            nlohmann::json{{"matrix_size", 4U},
                           {"angles", {0.25F, -0.5F}},
                           {"kernel_types", {"Rotation", "Reflection"}}});

    const sfFDN::TimeVaryingKroneckerFeedbackMatrixOptions time_varying_kronecker = {
        .matrix = kronecker,
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F},
                                {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F}},
    };
    REQUIRE(nlohmann::json(time_varying_kronecker) ==
            nlohmann::json{{"matrix", {{"matrix_size", 4U},
                                       {"angles", {0.25F, -0.5F}},
                                       {"kernel_types", {"Rotation", "Reflection"}}}},
                           {"time_varying_config",
                            {{{"frequency", 0.001F}, {"amplitude", 0.25F}, {"initial_phase", 0.5F}},
                             {{"frequency", 0.002F}, {"amplitude", -0.5F}, {"initial_phase", 0.75F}}}}});

    auto legacy_kronecker = nlohmann::json(kronecker);
    legacy_kronecker["time_varying_config"] = nlohmann::json::array();
    REQUIRE_THROWS(legacy_kronecker.get<sfFDN::KroneckerFeedbackMatrixOptions>());

    auto unknown_kronecker_key = nlohmann::json(kronecker);
    unknown_kronecker_key["unexpected"] = 1;
    REQUIRE_THROWS(unknown_kronecker_key.get<sfFDN::KroneckerFeedbackMatrixOptions>());

    const sfFDN::ChannelMatrixOptions channel_matrix = {
        .input_channel_count = 2U,
        .output_channel_count = 3U,
        .coefficients = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F},
    };
    REQUIRE(nlohmann::json(channel_matrix) == nlohmann::json{
                                                  {"input_channel_count", 2U},
                                                  {"output_channel_count", 3U},
                                                  {"coefficients", {1.F, 2.F, 3.F, 4.F, 5.F, 6.F}},
                                              });

    // An absent boundary matrix emits null rather than being omitted, so the shape does not depend on the value.
    const nlohmann::json mono_stage = sfFDN::InputStageConfig{};
    REQUIRE(mono_stage.size() == 4);
    REQUIRE(mono_stage.at("boundary_matrix").is_null());
}

TEST_CASE("FDNConfig JSON rejects representative malformed input", "[serialization]")
{
    const nlohmann::json valid = MakeRenderableConfig();
    std::vector<nlohmann::json> malformed;

    auto missing = valid;
    missing.erase("fdn_size");
    malformed.push_back(std::move(missing));

    auto wrong_type = valid;
    wrong_type["block_size"] = "sixteen";
    malformed.push_back(std::move(wrong_type));

    auto unknown_variant = valid;
    unknown_variant["input_block_config"]["single_channel_processors"] =
        nlohmann::json::array({{{"UnknownProcessorOptions", nlohmann::json::object()}}});
    malformed.push_back(std::move(unknown_variant));

    auto invalid_enum = valid;
    invalid_enum["delay_bank_config"]["interpolation_type"] = "Cubic";
    malformed.push_back(std::move(invalid_enum));

    auto null_number = valid;
    null_number["direct_gain"] = nullptr;
    malformed.push_back(std::move(null_number));

    auto unknown_stage_key = valid;
    unknown_stage_key["input_block_config"]["unexpected"] = 1;
    malformed.push_back(std::move(unknown_stage_key));

    auto too_many_stage_keys = nlohmann::json(MakeMimoConfig());
    too_many_stage_keys["output_block_config"]["unexpected"] = 1;
    malformed.push_back(std::move(too_many_stage_keys));

    auto malformed_channel_matrix = nlohmann::json(MakeMimoConfig());
    malformed_channel_matrix["direct_matrix"].erase("coefficients");
    malformed.push_back(std::move(malformed_channel_matrix));

    auto extra_channel_matrix_key = nlohmann::json(MakeMimoConfig());
    extra_channel_matrix_key["input_block_config"]["boundary_matrix"]["unexpected"] = 1;
    malformed.push_back(std::move(extra_channel_matrix_key));

    auto negative_channel_count = nlohmann::json(MakeMimoConfig());
    negative_channel_count["input_channel_count"] = -1;
    malformed.push_back(std::move(negative_channel_count));

    for (const auto& value : malformed)
    {
        REQUIRE_THROWS(value.get<sfFDN::FDNConfig>());
    }

    auto nullable_attenuation = valid;
    nullable_attenuation["attenuation_filter_bank_config"] = nullptr;
    REQUIRE_FALSE(nullable_attenuation.get<sfFDN::FDNConfig>().attenuation_filter_bank_config.has_value());

    const sfFDN::DelayOptions populated_delay = {
        .delay = 2.F,
        .max_delay = 4U,
        .interp_type = sfFDN::DelayInterpolationType::Linear,
        .lfo_config = sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = 0.25F, .initial_phase = 0.F},
    };
    auto absent_lfo = nlohmann::json(populated_delay);
    absent_lfo.erase("lfo_config");
    REQUIRE_FALSE(absent_lfo.get<sfFDN::DelayOptions>().lfo_config.has_value());
    auto null_lfo = nlohmann::json(populated_delay);
    null_lfo["lfo_config"] = nullptr;
    REQUIRE_FALSE(null_lfo.get<sfFDN::DelayOptions>().lfo_config.has_value());
}

TEST_CASE("JSON reads leave destinations unchanged on failure", "[serialization]")
{
    const auto config = MakeRenderableConfig();
    auto malformed_root = nlohmann::json(config);
    malformed_root["output_block_config"]["parallel_gains_config"]["gains"] = nlohmann::json::object();
    RequireUnchangedAfterFailedRead(malformed_root, config);

    const sfFDN::ScalarFeedbackMatrixOptions matrix = {
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = 2U,
                .generator = sfFDN::VariableDiffusionOptions{.diffusion = 0.5F},
                .rng_seed = 7U,
            },
    };
    auto malformed_matrix = nlohmann::json(matrix);
    malformed_matrix[nlohmann::json::json_pointer(
        "/source/GeneratedMatrixOptions/generator/VariableDiffusionOptions/diffusion")] = "late";
    RequireUnchangedAfterFailedRead(malformed_matrix, matrix);

    const sfFDN::KroneckerFeedbackMatrixOptions kronecker{
        .matrix_size = 4U,
        .angles = {0.25F, -0.5F},
        .kernel_types = {sfFDN::KroneckerKernelType::Rotation, sfFDN::KroneckerKernelType::Reflection},
    };
    auto malformed_kronecker = nlohmann::json(kronecker);
    malformed_kronecker["kernel_types"][1] = "Shear";
    RequireUnchangedAfterFailedRead(malformed_kronecker, kronecker);
}

TEST_CASE("MultichannelProcessorOptions serialization clears a reused destination", "[serialization]")
{
    const sfFDN::MultichannelProcessorOptions options{
        .channels = {sfFDN::FirOptions{.coeffs = {1.F}}, std::nullopt},
    };
    const nlohmann::json expected = options;
    nlohmann::json reused = {"stale", {"version", 1U}};

    sfFDN::to_json(reused, options);

    REQUIRE(reused == expected);
    REQUIRE(reused.get<sfFDN::MultichannelProcessorOptions>() == options);
}
