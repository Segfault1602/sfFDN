#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include "json_helper.h"
#include "rng.h"
#include "sffdn/sffdn.h"
#include "test_utils.h"

namespace
{

sfFDN::TimeVaryingFeedbackMatrixOptions MakeTimeVaryingMatrixOptions(uint32_t matrix_size)
{
    return {
        .matrix_size = matrix_size,
        .mode = sfFDN::TimeVaryingMatrixMode::Hadamard,
        .time_varying_config =
            {
                {.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.125F},
                {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F},
            },
    };
}

sfFDN::FDNConfig MakeTimeVaryingFDNConfig()
{
    sfFDN::FDNConfig config;
    config.fdn_size = 4;
    config.transposed = false;
    config.direct_gain = 0.F;
    config.block_size = 16;
    config.sample_rate = 48000.F;
    config.delay_bank_config = {
        .delays = {32.F, 37.F, 43.F, 47.F},
        .block_size = config.block_size,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.input_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Split,
        .gains = std::vector<float>(config.fdn_size, 0.5F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = MakeTimeVaryingMatrixOptions(config.fdn_size);
    config.output_block_config.parallel_gains_config = {
        .mode = sfFDN::ParallelGainsMode::Merge,
        .gains = std::vector<float>(config.fdn_size, 0.5F),
        .time_varying_config = {},
    };
    return config;
}

void RequireEqual(const sfFDN::TimeVaryingFeedbackMatrixOptions& actual,
                  const sfFDN::TimeVaryingFeedbackMatrixOptions& expected)
{
    REQUIRE(actual.matrix_size == expected.matrix_size);
    REQUIRE(actual.mode == expected.mode);
    REQUIRE(actual.rng_seed == expected.rng_seed);
    REQUIRE(actual.time_varying_config.size() == expected.time_varying_config.size());
    for (size_t index = 0; index < expected.time_varying_config.size(); ++index)
    {
        REQUIRE(actual.time_varying_config[index].frequency == expected.time_varying_config[index].frequency);
        REQUIRE(actual.time_varying_config[index].amplitude == expected.time_varying_config[index].amplitude);
        REQUIRE(actual.time_varying_config[index].initial_phase == expected.time_varying_config[index].initial_phase);
    }
}

void RequireEqual(const sfFDN::TimeVaryingSchroederAllpassSectionOptions& actual,
                  const sfFDN::TimeVaryingSchroederAllpassSectionOptions& expected)
{
    REQUIRE(actual.delays == expected.delays);
    REQUIRE(actual.gains == expected.gains);
    REQUIRE(actual.parallel == expected.parallel);
    REQUIRE(actual.time_varying_config.size() == expected.time_varying_config.size());
    for (size_t index = 0; index < expected.time_varying_config.size(); ++index)
    {
        REQUIRE(actual.time_varying_config[index].frequency == expected.time_varying_config[index].frequency);
        REQUIRE(actual.time_varying_config[index].amplitude == expected.time_varying_config[index].amplitude);
        REQUIRE(actual.time_varying_config[index].initial_phase == expected.time_varying_config[index].initial_phase);
    }
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
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1U, std::span(input).subspan(offset, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1U, std::span(output).subspan(offset, kBlockSize));
        std::fill(output.begin() + offset, output.begin() + offset + kBlockSize, 0.F);
        fdn.Process(input_buffer, output_buffer);
    }

    return output;
}

} // namespace

TEST_CASE("FDNConfig round-trips all configured processor options", "[serialization]")
{
    sfFDN::FDNConfig config;
    config.fdn_size = 4;
    config.transposed = false;
    config.direct_gain = 0.5f;
    config.block_size = 128;
    config.sample_rate = 48000;
    config.delay_bank_config = {
        {4, 7, 13, 23},
        128,
        sfFDN::DelayInterpolationType::None,
    };

    config.input_block_config.single_channel_processors = {
        sfFDN::AllpassFilterOptions{.coeff = 0.5f}, sfFDN::DelayOptions{.delay = 64},
        sfFDN::DattorroDelayOptions{.delay_config = {.delay = 96.f,
                                                     .max_delay = 256,
                                                     .interp_type = sfFDN::DelayInterpolationType::Allpass,
                                                     .lfo_config = sfFDN::ModulationOptions{.frequency = 0.0001f,
                                                                                            .amplitude = 8.f,
                                                                                            .initial_phase = 0.f}},
                                    .blend = 0.7071f,
                                    .feedforward = 1.f,
                                    .feedback = 0.7071f}};
    config.input_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Split, {0.5f, 0.3f, 0.4f, 0.8f}, {}};

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = 4,
        .type = sfFDN::ScalarMatrixType::Hadamard,
    };

    sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_config;
    for (size_t i = 0; i < 4; ++i)
    {
        attenuation_filter_bank_config.filter_configs.push_back(sfFDN::TwoBandFilterOptions{
            .t60s = {1.f, 0.5f},
            .delay = 64.f,
            .sample_rate = 48000.f,
        });
    }
    config.attenuation_filter_bank_config = attenuation_filter_bank_config;

    config.output_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Merge, {0.7f, 0.6f, 0.5f, 0.4f}, {}};

    sfFDN::MultichannelProcessorOptions dattorro_bank_config;
    dattorro_bank_config.channels.resize(4);
    for (size_t i = 0; i < dattorro_bank_config.channels.size(); ++i)
    {
        sfFDN::DattorroDelayOptions channel;
        channel.blend = 0.5f + (0.01f * static_cast<float>(i));
        channel.feedforward = 1.f;
        channel.feedback = 0.25f;
        channel.delay_config.delay = 64.f + (8.f * static_cast<float>(i));
        channel.delay_config.max_delay = 256;
        channel.delay_config.interp_type = sfFDN::DelayInterpolationType::Allpass;
        // Leave the last channel unmodulated, so that the optional lfo_config is exercised both ways.
        if (i + 1 < dattorro_bank_config.channels.size())
        {
            channel.delay_config.lfo_config = sfFDN::ModulationOptions{
                .frequency = 0.0002f, .amplitude = 4.f, .initial_phase = 0.25f * static_cast<float>(i)};
        }
        else
        {
            channel.delay_config.lfo_config = std::nullopt;
        }
        dattorro_bank_config.channels[i] = channel;
    }
    config.input_block_config.multichannel_processors = {dattorro_bank_config};

    // The shimmer nonlinearities. The single-channel ones are appended after the existing processors so that the
    // indices asserted below do not shift, and the multichannel banks go in the loop filter block, which is where
    // they belong in an FDN. Every bank leaves one channel null, so the optional entries are exercised both ways.
    config.input_block_config.single_channel_processors.emplace_back(sfFDN::ControllableFullWaveRectifierOptions{
        .alpha = 0.75f, .antialiasing = true, .dc_block = true, .sample_rate = 48000.f});
    config.input_block_config.single_channel_processors.emplace_back(
        sfFDN::SignalDependentFractionalDelayOptions{.d = 0.4f});
    config.input_block_config.single_channel_processors.emplace_back(
        sfFDN::RingModulatorOptions{.frequency = 0.002f, .amplitude = 1.4142f, .initial_phase = 0.375f});
    config.input_block_config.single_channel_processors.emplace_back(sfFDN::GraphicEQOptions{
        .gains_db = {},
        .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
        .sample_rate = 48000.F,
    });

    sfFDN::MultichannelProcessorOptions rectifier_bank_config;
    rectifier_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < rectifier_bank_config.channels.size(); ++i)
    {
        rectifier_bank_config.channels[i] =
            sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.1f * static_cast<float>(i + 1),
                                                        .antialiasing = (i % 2) == 0,
                                                        .dc_block = (i % 2) == 1,
                                                        .sample_rate = 48000.f};
    }

    sfFDN::MultichannelProcessorOptions sdfd_bank_config;
    sdfd_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < sdfd_bank_config.channels.size(); ++i)
    {
        sdfd_bank_config.channels[i] =
            sfFDN::SignalDependentFractionalDelayOptions{.d = 0.2f * static_cast<float>(i + 1)};
    }

    sfFDN::MultichannelProcessorOptions ring_mod_bank_config;
    ring_mod_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < ring_mod_bank_config.channels.size(); ++i)
    {
        ring_mod_bank_config.channels[i] = sfFDN::RingModulatorOptions{.frequency = 0.001f * static_cast<float>(i + 1),
                                                                       .amplitude = 1.4142f,
                                                                       .initial_phase = 0.25f * static_cast<float>(i)};
    }

    config.loop_filter_configs = {rectifier_bank_config, sdfd_bank_config, ring_mod_bank_config};

    const nlohmann::json j = config;

    sfFDN::FDNConfig deserialized_config = j.get<sfFDN::FDNConfig>();

    // Re-serializing the complete configuration catches fields that individual
    // assertions below do not happen to inspect.
    REQUIRE(nlohmann::json(deserialized_config) == j);

    const auto& single_channel_procs = deserialized_config.input_block_config.single_channel_processors;
    REQUIRE(single_channel_procs.size() == 7);
    REQUIRE(std::holds_alternative<sfFDN::DattorroDelayOptions>(single_channel_procs[2]));
    REQUIRE(std::holds_alternative<sfFDN::GraphicEQOptions>(single_channel_procs[6]));

    const auto& dattorro = std::get<sfFDN::DattorroDelayOptions>(single_channel_procs[2]);
    REQUIRE_THAT(dattorro.blend, Catch::Matchers::WithinAbs(0.7071f, 1e-5f));
    REQUIRE_THAT(dattorro.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-5f));
    REQUIRE_THAT(dattorro.feedback, Catch::Matchers::WithinAbs(0.7071f, 1e-5f));
    REQUIRE_THAT(dattorro.delay_config.delay, Catch::Matchers::WithinAbs(96.f, 1e-5f));
    REQUIRE(dattorro.delay_config.lfo_config.has_value());
    REQUIRE_THAT(dattorro.delay_config.lfo_config.value().frequency, Catch::Matchers::WithinAbs(0.0001f, 1e-8f));
    REQUIRE_THAT(dattorro.delay_config.lfo_config.value().amplitude, Catch::Matchers::WithinAbs(8.f, 1e-5f));
    REQUIRE_THAT(dattorro.delay_config.lfo_config.value().initial_phase, Catch::Matchers::WithinAbs(0.f, 1e-5f));

    const auto& multichannel_procs = deserialized_config.input_block_config.multichannel_processors;
    REQUIRE(multichannel_procs.size() == 1);
    REQUIRE(std::holds_alternative<sfFDN::MultichannelProcessorOptions>(multichannel_procs[0]));

    const auto& dattorro_bank = std::get<sfFDN::MultichannelProcessorOptions>(multichannel_procs[0]);
    REQUIRE(dattorro_bank.channels.size() == 4);
    for (size_t i = 0; i < dattorro_bank.channels.size(); ++i)
    {
        const auto& channel = std::get<sfFDN::DattorroDelayOptions>(dattorro_bank.channels[i].value());
        REQUIRE_THAT(channel.blend, Catch::Matchers::WithinAbs(0.5f + (0.01f * static_cast<float>(i)), 1e-5f));
        REQUIRE_THAT(channel.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-5f));
        REQUIRE_THAT(channel.feedback, Catch::Matchers::WithinAbs(0.25f, 1e-5f));
        REQUIRE_THAT(channel.delay_config.delay,
                     Catch::Matchers::WithinAbs(64.f + (8.f * static_cast<float>(i)), 1e-5f));
        REQUIRE(channel.delay_config.interp_type == sfFDN::DelayInterpolationType::Allpass);

        if (i + 1 < dattorro_bank.channels.size())
        {
            REQUIRE(channel.delay_config.lfo_config.has_value());
            REQUIRE_THAT(channel.delay_config.lfo_config->frequency, Catch::Matchers::WithinAbs(0.0002f, 1e-8f));
            REQUIRE_THAT(channel.delay_config.lfo_config->amplitude, Catch::Matchers::WithinAbs(4.f, 1e-5f));
            REQUIRE_THAT(channel.delay_config.lfo_config->initial_phase,
                         Catch::Matchers::WithinAbs(0.25f * static_cast<float>(i), 1e-5f));
        }
        else
        {
            REQUIRE_FALSE(channel.delay_config.lfo_config.has_value());
        }
    }

    // The shimmer nonlinearities, single channel.
    REQUIRE(std::holds_alternative<sfFDN::ControllableFullWaveRectifierOptions>(single_channel_procs[3]));
    const auto& rectifier = std::get<sfFDN::ControllableFullWaveRectifierOptions>(single_channel_procs[3]);
    REQUIRE_THAT(rectifier.alpha, Catch::Matchers::WithinAbs(0.75f, 1e-6f));
    REQUIRE(rectifier.antialiasing);
    REQUIRE(rectifier.dc_block);
    REQUIRE_THAT(rectifier.sample_rate, Catch::Matchers::WithinAbs(48000.f, 1e-3f));

    REQUIRE(std::holds_alternative<sfFDN::SignalDependentFractionalDelayOptions>(single_channel_procs[4]));
    REQUIRE_THAT(std::get<sfFDN::SignalDependentFractionalDelayOptions>(single_channel_procs[4]).d,
                 Catch::Matchers::WithinAbs(0.4f, 1e-6f));

    REQUIRE(std::holds_alternative<sfFDN::RingModulatorOptions>(single_channel_procs[5]));
    const auto& ring_mod = std::get<sfFDN::RingModulatorOptions>(single_channel_procs[5]);
    REQUIRE_THAT(ring_mod.frequency, Catch::Matchers::WithinAbs(0.002f, 1e-9f));
    REQUIRE_THAT(ring_mod.amplitude, Catch::Matchers::WithinAbs(1.4142f, 1e-6f));
    REQUIRE_THAT(ring_mod.initial_phase, Catch::Matchers::WithinAbs(0.375f, 1e-6f));

    // The shimmer nonlinearities, multichannel. The last channel of each bank is null and must stay null.
    const auto& loop_filters = deserialized_config.loop_filter_configs;
    REQUIRE(loop_filters.size() == 3);

    REQUIRE(std::holds_alternative<sfFDN::MultichannelProcessorOptions>(loop_filters[0]));
    const auto& rectifier_bank = std::get<sfFDN::MultichannelProcessorOptions>(loop_filters[0]);
    REQUIRE(rectifier_bank.channels.size() == 4);
    REQUIRE_FALSE(rectifier_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < rectifier_bank.channels.size(); ++i)
    {
        REQUIRE(rectifier_bank.channels[i].has_value());
        const auto& channel = std::get<sfFDN::ControllableFullWaveRectifierOptions>(rectifier_bank.channels[i].value());
        REQUIRE_THAT(channel.alpha, Catch::Matchers::WithinAbs(0.1f * static_cast<float>(i + 1), 1e-6f));
        REQUIRE(channel.antialiasing == ((i % 2) == 0));
        REQUIRE(channel.dc_block == ((i % 2) == 1));
    }

    REQUIRE(std::holds_alternative<sfFDN::MultichannelProcessorOptions>(loop_filters[1]));
    const auto& sdfd_bank = std::get<sfFDN::MultichannelProcessorOptions>(loop_filters[1]);
    REQUIRE(sdfd_bank.channels.size() == 4);
    REQUIRE_FALSE(sdfd_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < sdfd_bank.channels.size(); ++i)
    {
        REQUIRE(sdfd_bank.channels[i].has_value());
        REQUIRE_THAT(std::get<sfFDN::SignalDependentFractionalDelayOptions>(sdfd_bank.channels[i].value()).d,
                     Catch::Matchers::WithinAbs(0.2f * static_cast<float>(i + 1), 1e-6f));
    }

    REQUIRE(std::holds_alternative<sfFDN::MultichannelProcessorOptions>(loop_filters[2]));
    const auto& ring_mod_bank = std::get<sfFDN::MultichannelProcessorOptions>(loop_filters[2]);
    REQUIRE(ring_mod_bank.channels.size() == 4);
    REQUIRE_FALSE(ring_mod_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < ring_mod_bank.channels.size(); ++i)
    {
        REQUIRE(ring_mod_bank.channels[i].has_value());
        const auto& channel = std::get<sfFDN::RingModulatorOptions>(ring_mod_bank.channels[i].value());
        REQUIRE_THAT(channel.frequency, Catch::Matchers::WithinAbs(0.001f * static_cast<float>(i + 1), 1e-9f));
        REQUIRE_THAT(channel.initial_phase, Catch::Matchers::WithinAbs(0.25f * static_cast<float>(i), 1e-6f));
    }

    // The configuration must also survive being turned into an actual FDN.
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(deserialized_config));
}

TEST_CASE("FDNConfig JSON round-trips defaults, optionals, and variants exactly", "[serialization]")
{
    auto defaults_and_absent_optionals = MakeTimeVaryingFDNConfig();

    auto populated_optionals_and_variants = MakeTimeVaryingFDNConfig();
    populated_optionals_and_variants.direct_gain = 0.25F;
    populated_optionals_and_variants.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = 4U,
        .type = sfFDN::ScalarMatrixType::Random,
        .custom_matrix =
            std::vector<float>{1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F},
        .rng_seed = 42U,
        .arg = 0.75F,
    };
    populated_optionals_and_variants.input_block_config.single_channel_processors.emplace_back(sfFDN::DelayOptions{
        .delay = 12.5F,
        .max_delay = 32U,
        .interp_type = sfFDN::DelayInterpolationType::Linear,
        .lfo_config = sfFDN::ModulationOptions{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F}});
    populated_optionals_and_variants.output_block_config.single_channel_processors.emplace_back(
        sfFDN::AllpassFilterOptions{.coeff = -0.4F});

    const std::vector<std::pair<const char*, sfFDN::FDNConfig>> cases = {
        {"defaults and absent optionals", defaults_and_absent_optionals},
        {"populated optionals and processor variants", populated_optionals_and_variants},
    };

    for (const auto& [name, original] : cases)
    {
        DYNAMIC_SECTION(name)
        {
            const nlohmann::json serialized = original;
            const auto round_tripped = serialized.get<sfFDN::FDNConfig>();
            REQUIRE(nlohmann::json(round_tripped) == serialized);
            REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(round_tripped));
        }
    }
}

TEST_CASE("FDNConfig JSON rejects malformed required fields and variants", "[serialization]")
{
    const nlohmann::json valid = MakeTimeVaryingFDNConfig();

    SECTION("missing required fields")
    {
        for (const auto* field : {"fdn_size", "delay_bank_config", "feedback_matrix_config"})
        {
            auto malformed = valid;
            malformed.erase(field);
            REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
        }
    }

    SECTION("wrong field types and non-finite values represented as null")
    {
        auto wrong_type = valid;
        wrong_type["block_size"] = "sixteen";
        REQUIRE_THROWS(wrong_type.get<sfFDN::FDNConfig>());

        auto non_finite = valid;
        non_finite["direct_gain"] = std::numeric_limits<float>::infinity();
        const auto reparsed = nlohmann::json::parse(non_finite.dump());
        REQUIRE(reparsed["direct_gain"].is_null());
        REQUIRE_THROWS(reparsed.get<sfFDN::FDNConfig>());
    }

    SECTION("empty, unknown, and multiple unknown processor tags")
    {
        auto empty_variant = valid;
        empty_variant["input_block_config"]["single_channel_processors"] = nlohmann::json::array({nlohmann::json{}});
        REQUIRE_THROWS(empty_variant.get<sfFDN::FDNConfig>());

        auto unknown_variant = valid;
        unknown_variant["input_block_config"]["single_channel_processors"] =
            nlohmann::json::array({{{"UnknownProcessorOptions", nlohmann::json::object()}}});
        REQUIRE_THROWS(unknown_variant.get<sfFDN::FDNConfig>());

        auto multiple_tags = valid;
        multiple_tags["input_block_config"]["single_channel_processors"] =
            nlohmann::json::array({{{"UnknownProcessorOptions", nlohmann::json::object()},
                                    {"AnotherUnknownProcessorOptions", nlohmann::json::object()}}});
        REQUIRE_THROWS(multiple_tags.get<sfFDN::FDNConfig>());
    }

    SECTION("invalid enum strings")
    {
        auto invalid_enum = valid;
        invalid_enum["delay_bank_config"]["interpolation_type"] = "Cubic";
        const auto deserialized = invalid_enum.get<sfFDN::FDNConfig>();
        REQUIRE(deserialized.delay_bank_config.interpolation_type == sfFDN::DelayInterpolationType::None);
    }
}

TEST_CASE("FDNConfig rejects invalid processor graphs during construction", "[serialization]")
{
    auto invalid_parallel_mode = MakeTimeVaryingFDNConfig();
    invalid_parallel_mode.loop_filter_configs.emplace_back(sfFDN::ParallelGainsOptions{
        .mode = sfFDN::ParallelGainsMode::Split, .gains = std::vector<float>(4U, 1.F), .time_varying_config = {}});
    REQUIRE_THROWS(sfFDN::CreateFDNFromConfig(invalid_parallel_mode));

    auto invalid_channel_count = MakeTimeVaryingFDNConfig();
    invalid_channel_count.loop_filter_configs.emplace_back(sfFDN::MultichannelProcessorOptions{
        .channels = {sfFDN::RingModulatorOptions{.frequency = 0.001F, .amplitude = 1.F, .initial_phase = 0.F}}});
    REQUIRE_THROWS(sfFDN::CreateFDNFromConfig(invalid_channel_count));

    auto invalid_custom_matrix = MakeTimeVaryingFDNConfig();
    invalid_custom_matrix.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = 4U, .type = sfFDN::ScalarMatrixType::Random, .custom_matrix = std::vector<float>{1.F, 0.F, 0.F}};
    REQUIRE_THROWS(sfFDN::CreateFDNFromConfig(invalid_custom_matrix));
}

TEST_CASE("FDNConfig accepts generic banks in every multichannel placement", "[serialization]")
{
    for (const bool transposed : {false, true})
    {
        const sfFDN::MultichannelProcessorOptions bank{
            .channels = {sfFDN::FirOptions{.coeffs = {0.5F, 0.25F, -0.125F}},
                         sfFDN::AllpassFilterOptions{.coeff = -0.4F},
                         sfFDN::DelayOptions{.delay = 3.F, .max_delay = 8U},
                         sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5F}},
        };
        for (const char* placement : {"input", "loop", "output"})
        {
            auto config = MakeTimeVaryingFDNConfig();
            config.transposed = transposed;
            if (std::string_view(placement) == "input")
            {
                config.input_block_config.multichannel_processors.emplace_back(bank);
            }
            else if (std::string_view(placement) == "loop")
            {
                config.loop_filter_configs.emplace_back(bank);
            }
            else
            {
                config.output_block_config.multichannel_processors.emplace_back(bank);
            }

            const auto rendered = RenderFDN(*sfFDN::CreateFDNFromConfig(config));
            auto no_bank = MakeTimeVaryingFDNConfig();
            no_bank.transposed = transposed;
            const auto baseline = RenderFDN(*sfFDN::CreateFDNFromConfig(no_bank));
            REQUIRE(rendered != baseline);

            auto invalid = config;
            if (std::string_view(placement) == "input")
            {
                std::get<sfFDN::MultichannelProcessorOptions>(invalid.input_block_config.multichannel_processors[0])
                    .channels.pop_back();
            }
            else if (std::string_view(placement) == "loop")
            {
                std::get<sfFDN::MultichannelProcessorOptions>(invalid.loop_filter_configs[0])
                    .channels.emplace_back(sfFDN::FirOptions{.coeffs = {1.F}});
            }
            else
            {
                std::get<sfFDN::MultichannelProcessorOptions>(invalid.output_block_config.multichannel_processors[0])
                    .channels.clear();
            }
            REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(invalid), std::runtime_error);
        }
    }
}

TEST_CASE("MultichannelProcessorOptions JSON rejects legacy and malformed forms", "[serialization]")
{
    for (const char* legacy :
         {"MultichannelSchroederAllpassSectionOptions", "MultichannelTimeVaryingSchroederAllpassSectionOptions",
          "MultichannelDattorroDelayOptions", "MultichannelFirOptions",
          "MultichannelControllableFullWaveRectifierOptions", "MultichannelSignalDependentFractionalDelayOptions",
          "MultichannelRingModulatorOptions"})
    {
        const nlohmann::json old = {{legacy, nlohmann::json::object()}};
        REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(old));
    }

    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(
        {{"MultichannelProcessorOptions", {{"channels", nlohmann::json::object()}}}}));
    REQUIRE_THROWS(nlohmann::json{
        {"MultichannelProcessorOptions", {{"channels", nlohmann::json::array({nlohmann::json::object()})}}}}
                       .at("MultichannelProcessorOptions")
                       .get<sfFDN::MultichannelProcessorOptions>());
    REQUIRE_THROWS(nlohmann::json{
        {"MultichannelProcessorOptions", {{"channels", nlohmann::json::array({{{"UnknownOptions", {}}}})}}}}
                       .at("MultichannelProcessorOptions")
                       .get<sfFDN::MultichannelProcessorOptions>());
    REQUIRE_THROWS(nlohmann::json{
        {"MultichannelProcessorOptions",
         {{"channels",
           nlohmann::json::array({{{"FirOptions", {{"coeffs", {1.F}}}},
                                   {"DelayOptions", {{"delay", 2.F}, {"max_delay", 4U}, {"interp_type", "None"}}}}})}}}}
                       .at("MultichannelProcessorOptions")
                       .get<sfFDN::MultichannelProcessorOptions>());
}

TEST_CASE("MultichannelProcessorOptions round-trips every single-channel option", "[serialization]")
{
    const sfFDN::MultichannelProcessorOptions options{
        .channels =
            {
                sfFDN::SchroederAllpassSectionOptions{.delays = {2.F}, .gains = {0.25F}},
                sfFDN::TimeVaryingSchroederAllpassSectionOptions{
                    .delays = {3.F},
                    .gains = {0.25F},
                    .time_varying_config = {{.frequency = 0.01F, .amplitude = 0.1F, .initial_phase = 0.F}}},
                sfFDN::AllpassFilterOptions{.coeff = 0.25F},
                sfFDN::CascadedBiquadsOptions{.coeffs = {{1.F, 0.F, 0.F, 1.F, 0.F, 0.F}}},
                sfFDN::FirOptions{.coeffs = {1.F}},
                sfFDN::DelayOptions{.delay = 2.F, .max_delay = 4U},
                sfFDN::DelayOptions{
                    .delay = 3.F,
                    .max_delay = 5U,
                    .interp_type = sfFDN::DelayInterpolationType::Linear,
                    .lfo_config =
                        sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = 0.25F, .initial_phase = 0.F}},
                sfFDN::GraphicEQOptions{
                    .gains_db = {},
                    .freqs = {32.F, 64.F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
                    .sample_rate = 48000.F},
                sfFDN::DattorroDelayOptions{.delay_config = {.delay = 4.F, .max_delay = 8U}},
                sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.5F, .dc_block = false},
                sfFDN::SignalDependentFractionalDelayOptions{.d = 0.5F},
                sfFDN::RingModulatorOptions{.frequency = 0.01F, .amplitude = 1.F, .initial_phase = 0.F},
                std::nullopt,
            },
    };

    const nlohmann::json serialized = options;
    const auto round_tripped = serialized.get<sfFDN::MultichannelProcessorOptions>();
    REQUIRE(nlohmann::json(round_tripped) == serialized);
    REQUIRE_FALSE(round_tripped.channels.back().has_value());

    nlohmann::json reused = {{"stale_wrapper", {{"version", 1U}}}};
    sfFDN::to_json(reused, options);
    REQUIRE(reused == serialized);
    REQUIRE_NOTHROW(reused.get<sfFDN::MultichannelProcessorOptions>());
}

TEST_CASE("JSON readers preserve established metadata tolerance outside generic banks", "[serialization]")
{
    const nlohmann::json single_channel = {
        {"FirOptions", {{"coeffs", {0.5F, -0.25F}}}},
        {"metadata", {{"source", "preset"}}},
    };
    const auto decoded_single = sfFDN::SingleChannelProcessorFromJson(single_channel);
    REQUIRE(std::holds_alternative<sfFDN::FirOptions>(decoded_single));
    REQUIRE(std::get<sfFDN::FirOptions>(decoded_single).coeffs == std::vector<float>{0.5F, -0.25F});

    const nlohmann::json retained_multichannel = {
        {"ParallelGainsConfig",
         {{"mode", "Parallel"}, {"gains", {0.5F}}, {"time_varying_config", nlohmann::json::array()}}},
        {"metadata", {{"source", "preset"}}},
    };
    const auto decoded_multichannel = sfFDN::MultichannelProcessorFromJson(retained_multichannel);
    REQUIRE(std::holds_alternative<sfFDN::ParallelGainsOptions>(decoded_multichannel));
    REQUIRE(std::get<sfFDN::ParallelGainsOptions>(decoded_multichannel).gains == std::vector<float>{0.5F});

    const nlohmann::json generic_multichannel = {
        {"MultichannelProcessorOptions", {{"channels", nlohmann::json::array()}}},
        {"metadata", {{"source", "preset"}}},
    };
    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(generic_multichannel));

    const nlohmann::json generic_channels = {
        {"channels", nlohmann::json::array()},
        {"metadata", {{"source", "preset"}}},
    };
    REQUIRE_THROWS(generic_channels.get<sfFDN::MultichannelProcessorOptions>());
}

TEST_CASE("Multichannel processor alternatives use canonical JSON forms", "[serialization]")
{
    const nlohmann::json parallel_gains = {
        {"ParallelGainsConfig",
         {{"mode", "Parallel"},
          {"gains", {0.25F, -0.5F}},
          {"time_varying_config", {{{"frequency", 0.001F}, {"amplitude", 0.25F}, {"initial_phase", 0.5F}}}}}},
    };
    const nlohmann::json attenuation = {
        {"AttenuationFilterBankOptions",
         {{{"TwoBandFilterConfig", {{"t60s", {1.F, 0.5F}}, {"delay", 12.F}, {"sample_rate", 48000.F}}}}}},
    };
    const nlohmann::json delay_bank = {
        {"DelayBankOptions", {{"delays", {3.F, 5.F}}, {"block_size", 16U}, {"interpolation_type", "Linear"}}},
    };
    const nlohmann::json time_varying_delay_bank = {
        {"DelayBankTimeVaryingOptions",
         {{"delays", {3.F, 5.F}},
          {"max_delay", 8U},
          {"interpolation_type", "Linear"},
          {"time_varying_config",
           {{{"frequency", 0.001F}, {"amplitude", 0.25F}, {"initial_phase", 0.5F}},
            {{"frequency", 0.002F}, {"amplitude", -0.25F}, {"initial_phase", 0.25F}}}}}},
    };
    const nlohmann::json cascaded_matrix = {
        {"CascadedFeedbackMatrixInfo",
         {{"matrix_size", 4U},
          {"stage_count", 2U},
          {"sparsity", 3.F},
          {"type", "Hadamard"},
          {"gain_per_samples", 0.8F}}},
    };
    const nlohmann::json scalar_matrix = {
        {"ScalarFeedbackMatrixOptions",
         {{"matrix_size", 2U},
          {"type", "VariableDiffusion"},
          {"custom_matrix", {1.F, 0.F, 0.F, 1.F}},
          {"rng_seed", 17U},
          {"arg", 0.75F}}},
    };

    for (const auto& fixture :
         {parallel_gains, attenuation, delay_bank, time_varying_delay_bank, cascaded_matrix, scalar_matrix})
    {
        REQUIRE(fixture.is_object());
        REQUIRE(fixture.size() == 1U);
        const auto decoded = sfFDN::MultichannelProcessorFromJson(fixture);
        const nlohmann::json serialized = sfFDN::ToJson(decoded);
        REQUIRE(serialized == fixture);
    }

    const auto parallel = sfFDN::MultichannelProcessorFromJson(parallel_gains);
    REQUIRE(std::holds_alternative<sfFDN::ParallelGainsOptions>(parallel));
    REQUIRE(std::get<sfFDN::ParallelGainsOptions>(parallel).gains == std::vector<float>{0.25F, -0.5F});

    const auto attenuation_bank = sfFDN::MultichannelProcessorFromJson(attenuation);
    REQUIRE(std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(attenuation_bank));
    REQUIRE(std::holds_alternative<sfFDN::TwoBandFilterOptions>(
        std::get<sfFDN::AttenuationFilterBankOptions>(attenuation_bank).filter_configs[0]));

    const auto delays = sfFDN::MultichannelProcessorFromJson(delay_bank);
    REQUIRE(std::holds_alternative<sfFDN::DelayBankOptions>(delays));
    REQUIRE(std::get<sfFDN::DelayBankOptions>(delays).interpolation_type == sfFDN::DelayInterpolationType::Linear);

    const auto time_varying_delays = sfFDN::MultichannelProcessorFromJson(time_varying_delay_bank);
    REQUIRE(std::holds_alternative<sfFDN::DelayBankTimeVaryingOptions>(time_varying_delays));
    REQUIRE(std::get<sfFDN::DelayBankTimeVaryingOptions>(time_varying_delays).time_varying_config.size() == 2U);

    const auto cascaded = sfFDN::MultichannelProcessorFromJson(cascaded_matrix);
    REQUIRE(std::holds_alternative<sfFDN::CascadedFeedbackMatrixOptions>(cascaded));
    REQUIRE(std::get<sfFDN::CascadedFeedbackMatrixOptions>(cascaded).stage_count == 2U);

    const auto scalar = sfFDN::MultichannelProcessorFromJson(scalar_matrix);
    REQUIRE(std::holds_alternative<sfFDN::ScalarFeedbackMatrixOptions>(scalar));
    REQUIRE(std::get<sfFDN::ScalarFeedbackMatrixOptions>(scalar).custom_matrix.has_value());
}

TEST_CASE("FDNConfig JSON round-trip preserves rendered output", "[serialization]")
{
    const auto config = MakeTimeVaryingFDNConfig();
    const auto round_tripped = nlohmann::json(config).get<sfFDN::FDNConfig>();
    const auto original_fdn = sfFDN::CreateFDNFromConfig(config);
    const auto round_tripped_fdn = sfFDN::CreateFDNFromConfig(round_tripped);

    std::vector<float> input(16U * 8U, 0.F);
    std::vector<float> original_output(input.size(), 0.F);
    std::vector<float> round_tripped_output(input.size(), 0.F);
    input[0] = 1.F;

    for (size_t block = 0; block < 8U; ++block)
    {
        sfFDN::AudioBuffer const input_buffer(16U, 1U, std::span(input).subspan(block * 16U, 16U));
        sfFDN::AudioBuffer original_output_buffer(16U, 1U, std::span(original_output).subspan(block * 16U, 16U));
        sfFDN::AudioBuffer round_tripped_output_buffer(16U, 1U,
                                                       std::span(round_tripped_output).subspan(block * 16U, 16U));
        original_fdn->Process(input_buffer, original_output_buffer);
        round_tripped_fdn->Process(input_buffer, round_tripped_output_buffer);
    }

    REQUIRE(original_output == round_tripped_output);
}

TEST_CASE("TimeVaryingFeedbackMatrixOptions round-trips through JSON", "[serialization]")
{
    const auto options = MakeTimeVaryingMatrixOptions(4);

    nlohmann::json json = options;
    const auto deserialized_options = json.get<sfFDN::TimeVaryingFeedbackMatrixOptions>();

    RequireEqual(deserialized_options, options);
}

TEST_CASE("TimeVaryingFeedbackMatrixOptions reproduces RealSchur processing after JSON round-trip", "[serialization]")
{
    constexpr uint32_t kOrder = 6U;
    const sfFDN::TimeVaryingFeedbackMatrixOptions options = {
        .matrix_size = kOrder,
        .mode = sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config =
            {
                {.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.125F},
                {.frequency = 0.002F, .amplitude = -0.5F, .initial_phase = 0.75F},
                {.frequency = 0.003F, .amplitude = 0.7F, .initial_phase = 0.25F},
            },
    };
    const nlohmann::json json = options;
    const auto deserialized_options = json.get<sfFDN::TimeVaryingFeedbackMatrixOptions>();
    RequireEqual(deserialized_options, options);

    sfFDN::TimeVaryingFeedbackMatrix original(options);
    sfFDN::TimeVaryingFeedbackMatrix round_tripped(deserialized_options);
    std::vector<float> input(kOrder * 64U, 0.0F);
    std::vector<float> original_output(input.size(), 0.0F);
    std::vector<float> round_tripped_output(input.size(), 0.0F);
    input[0] = 1.0F;
    const sfFDN::AudioBuffer input_buffer(64U, kOrder, input);
    sfFDN::AudioBuffer original_output_buffer(64U, kOrder, original_output);
    sfFDN::AudioBuffer round_tripped_output_buffer(64U, kOrder, round_tripped_output);
    original.Process(input_buffer, original_output_buffer);
    round_tripped.Process(input_buffer, round_tripped_output_buffer);

    REQUIRE(original_output == round_tripped_output);
}

TEST_CASE("FDNConfig serializes a time-varying feedback matrix", "[serialization]")
{
    const auto config = MakeTimeVaryingFDNConfig();

    nlohmann::json json = config;
    const auto deserialized_config = json.get<sfFDN::FDNConfig>();

    REQUIRE(
        std::holds_alternative<sfFDN::TimeVaryingFeedbackMatrixOptions>(deserialized_config.feedback_matrix_config));
    REQUIRE(json["feedback_matrix_config"].contains("TimeVaryingFeedbackMatrixOptions"));
    RequireEqual(std::get<sfFDN::TimeVaryingFeedbackMatrixOptions>(deserialized_config.feedback_matrix_config),
                 std::get<sfFDN::TimeVaryingFeedbackMatrixOptions>(config.feedback_matrix_config));
}

TEST_CASE("FDNConfig creates an FDN with a time-varying feedback matrix", "[serialization]")
{
    const auto config = MakeTimeVaryingFDNConfig();
    const auto fdn = sfFDN::CreateFDNFromConfig(config);

    REQUIRE(dynamic_cast<sfFDN::TimeVaryingFeedbackMatrix*>(fdn->GetFeedbackMatrix()) != nullptr);

    std::vector<float> input(64, 0.F);
    std::vector<float> output(input.size(), 0.F);
    input[0] = 1.F;
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    fdn->Process(input_buffer, output_buffer);

    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
}

TEST_CASE("FDNConfig rejects invalid time-varying feedback matrix sizes", "[serialization]")
{
    auto mismatched_config = MakeTimeVaryingFDNConfig();
    mismatched_config.feedback_matrix_config = MakeTimeVaryingMatrixOptions(8);
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(mismatched_config), std::runtime_error);

    auto non_power_of_two_config = MakeTimeVaryingFDNConfig();
    non_power_of_two_config.fdn_size = 6;
    non_power_of_two_config.delay_bank_config.delays = {32.F, 37.F, 43.F, 47.F, 53.F, 59.F};
    non_power_of_two_config.input_block_config.parallel_gains_config.gains.assign(6, 0.5F);
    non_power_of_two_config.output_block_config.parallel_gains_config.gains.assign(6, 0.5F);
    non_power_of_two_config.feedback_matrix_config = MakeTimeVaryingMatrixOptions(6);
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(non_power_of_two_config), std::runtime_error);

    auto real_schur_config = non_power_of_two_config;
    auto real_schur_options = MakeTimeVaryingMatrixOptions(6);
    real_schur_options.mode = sfFDN::TimeVaryingMatrixMode::RealSchur;
    real_schur_options.rng_seed = 0x5EED1234U;
    real_schur_options.time_varying_config.clear();
    real_schur_config.feedback_matrix_config = real_schur_options;
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(real_schur_config));
}

TEST_CASE("FDNConfig rejects invalid time-varying feedback matrix options before construction", "[serialization]")
{
    auto wrong_modulation_count = MakeTimeVaryingFDNConfig();
    wrong_modulation_count.fdn_size = 8U;
    wrong_modulation_count.delay_bank_config.delays = {32.F, 37.F, 43.F, 47.F, 53.F, 59.F, 61.F, 67.F};
    wrong_modulation_count.input_block_config.parallel_gains_config.gains.assign(8U, 0.5F);
    wrong_modulation_count.output_block_config.parallel_gains_config.gains.assign(8U, 0.5F);
    wrong_modulation_count.feedback_matrix_config = MakeTimeVaryingMatrixOptions(8U);
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(wrong_modulation_count), std::runtime_error);

    nlohmann::json matrix_json = MakeTimeVaryingMatrixOptions(4U);
    matrix_json["mode"] = "Count";
    const auto sentinel_mode_options = matrix_json.get<sfFDN::TimeVaryingFeedbackMatrixOptions>();
    REQUIRE(sentinel_mode_options.mode == sfFDN::TimeVaryingMatrixMode::Count);

    auto sentinel_mode = MakeTimeVaryingFDNConfig();
    sentinel_mode.feedback_matrix_config = sentinel_mode_options;
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(sentinel_mode), std::runtime_error);
}

TEST_CASE("MultichannelProcessorOptions round-trips time-varying allpass banks through JSON", "[serialization]")
{
    const sfFDN::TimeVaryingSchroederAllpassSectionOptions section{
        .delays = {7.F, 13.F},
        .gains = {0.4F, -0.3F},
        .time_varying_config =
            {
                {.frequency = 0.001F, .amplitude = 0.2F, .initial_phase = 0.125F},
                {.frequency = 0.002F, .amplitude = -0.1F, .initial_phase = 0.75F},
            },
        .parallel = true,
    };
    const sfFDN::MultichannelProcessorOptions bank{
        .channels = {section, std::nullopt, section},
    };

    const nlohmann::json section_json = section;
    const nlohmann::json bank_json = bank;
    RequireEqual(section_json.get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(), section);

    const auto round_tripped_bank = bank_json.get<sfFDN::MultichannelProcessorOptions>();
    REQUIRE(round_tripped_bank.channels.size() == bank.channels.size());
    REQUIRE_FALSE(round_tripped_bank.channels[1].has_value());
    for (const size_t index : {0U, 2U})
    {
        RequireEqual(
            std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(round_tripped_bank.channels[index].value()),
            std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(bank.channels[index].value()));
    }
}

TEST_CASE("FDNConfig serializes and creates time-varying Schroeder allpasses", "[serialization]")
{
    auto config = MakeTimeVaryingFDNConfig();
    const sfFDN::TimeVaryingSchroederAllpassSectionOptions input_section{
        .delays = {5.F},
        .gains = {0.4F},
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.2F, .initial_phase = 0.25F}},
    };
    config.input_block_config.single_channel_processors.emplace_back(input_section);

    sfFDN::MultichannelProcessorOptions loop_bank;
    for (uint32_t channel = 0; channel < config.fdn_size; ++channel)
    {
        loop_bank.channels.emplace_back(sfFDN::TimeVaryingSchroederAllpassSectionOptions{
            .delays = {7.F + static_cast<float>(channel)},
            .gains = {0.35F},
            .time_varying_config = {{.frequency = 0.0005F * static_cast<float>(channel + 1U),
                                     .amplitude = 0.2F,
                                     .initial_phase =
                                         static_cast<float>(channel) / static_cast<float>(config.fdn_size)}},
        });
    }
    config.loop_filter_configs.emplace_back(loop_bank);

    const nlohmann::json json = config;
    REQUIRE(json["input_block_config"]["single_channel_processors"][0].contains(
        "TimeVaryingSchroederAllpassSectionOptions"));
    REQUIRE(json["loop_filter_configs"][0].contains("MultichannelProcessorOptions"));

    const auto round_tripped = json.get<sfFDN::FDNConfig>();
    REQUIRE(std::holds_alternative<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(
        round_tripped.input_block_config.single_channel_processors[0]));
    REQUIRE(std::holds_alternative<sfFDN::MultichannelProcessorOptions>(round_tripped.loop_filter_configs[0]));
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(round_tripped));

    auto invalid = round_tripped;
    auto& invalid_section = std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(
        invalid.input_block_config.single_channel_processors[0]);
    invalid_section.gains[0] = 1.F;
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(invalid), std::runtime_error);
}

TEST_CASE("ScalarFeedbackMatrixOptions JSON round-trip preserves row-major custom matrix", "[serialization]")
{
    // A non-symmetric 3x3 matrix in row-major order: flat[row*N+col] = A[row,col].
    constexpr uint32_t N = 3;
    const std::vector<float> kMatrix = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

    sfFDN::ScalarFeedbackMatrixOptions original;
    original.matrix_size = N;
    original.type = sfFDN::ScalarMatrixType::Random;
    original.custom_matrix = kMatrix;

    const nlohmann::json j = original;
    const auto deserialized = j.get<sfFDN::ScalarFeedbackMatrixOptions>();

    REQUIRE(deserialized.matrix_size == N);
    REQUIRE(deserialized.custom_matrix.has_value());
    REQUIRE(deserialized.custom_matrix->size() == N * N);
    for (size_t i = 0; i < kMatrix.size(); ++i)
    {
        REQUIRE((*deserialized.custom_matrix)[i] == kMatrix[i]);
    }

    // Constructing from the deserialized options must produce a matrix with the same
    // row-major coefficients: GetCoefficient(row,col) == kMatrix[row*N+col].
    sfFDN::ScalarFeedbackMatrix mat(deserialized);
    for (auto row = 0u; row < N; ++row)
    {
        for (auto col = 0u; col < N; ++col)
        {
            REQUIRE_THAT(mat.GetCoefficient(row, col), Catch::Matchers::WithinAbs(kMatrix[row * N + col], 0.f));
        }
    }
}
