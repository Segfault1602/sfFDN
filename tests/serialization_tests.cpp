#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string_view>
#include <vector>

#include "json_helper.h"
#include "rng.h"
#include "sffdn/sffdn.h"
#include "test_utils.h"
#include <sffdn/serialization.h>

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
        .gains = std::vector<float>(config.fdn_size, 0.5F),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = MakeTimeVaryingMatrixOptions(config.fdn_size);
    config.output_block_config.parallel_gains_config = {
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

template <typename Options>
void RequireUnchangedAfterFailedRead(const nlohmann::json& malformed, Options options)
{
    const nlohmann::json before = options;
    REQUIRE_THROWS(malformed.get_to(options));
    REQUIRE(nlohmann::json(options) == before);
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
        {128, 131, 137, 149},
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
    config.input_block_config.parallel_gains_config = {.gains = {0.5f, 0.3f, 0.4f, 0.8f}, .time_varying_config = {}};

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

    config.output_block_config.parallel_gains_config = {.gains = {0.7f, 0.6f, 0.5f, 0.4f}, .time_varying_config = {}};

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
    std::unique_ptr<sfFDN::FDN> fdn;
    REQUIRE_NOTHROW(fdn = sfFDN::CreateFDNFromConfig(deserialized_config));
    REQUIRE(fdn->GetDelayBank().GetDelays() == deserialized_config.delay_bank_config.delays);
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
        for (const auto* field :
             {"fdn_size", "delay_bank_config", "feedback_matrix_config", "attenuation_filter_bank_config"})
        {
            auto malformed = valid;
            malformed.erase(field);
            REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
        }

        auto nullable_attenuation = valid;
        nullable_attenuation["attenuation_filter_bank_config"] = nullptr;
        REQUIRE_NOTHROW(nullable_attenuation.get<sfFDN::FDNConfig>());
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
        REQUIRE_THROWS(invalid_enum.get<sfFDN::FDNConfig>());
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

TEST_CASE("JSON readers reject metadata wrapper siblings", "[serialization]")
{
    const nlohmann::json single_channel = {
        {"FirOptions", {{"coeffs", {0.5F, -0.25F}}}},
        {"metadata", {{"source", "preset"}}},
    };
    REQUIRE_THROWS(sfFDN::SingleChannelProcessorFromJson(single_channel));

    const nlohmann::json multichannel = {
        {"ParallelGainsConfig",
         {{"mode", "Parallel"}, {"gains", {0.5F}}, {"time_varying_config", nlohmann::json::array()}}},
        {"metadata", {{"source", "preset"}}},
    };
    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(multichannel));

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
          {"gain_per_samples", 0.8F},
          {"rng_seed", 0x5EED1234U}}},
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
    REQUIRE(std::get<sfFDN::CascadedFeedbackMatrixOptions>(cascaded).rng_seed == 0x5EED1234U);

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

TEST_CASE("FDNConfig JSON preserves seeded scalar and cascaded feedback matrices", "[serialization]")
{
    auto scalar_config = MakeTimeVaryingFDNConfig();
    scalar_config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = scalar_config.fdn_size,
        .type = sfFDN::ScalarMatrixType::Random,
        .rng_seed = 0x1234ABCDU,
    };
    const auto scalar_round_tripped = nlohmann::json(scalar_config).get<sfFDN::FDNConfig>();
    const auto& scalar_options =
        std::get<sfFDN::ScalarFeedbackMatrixOptions>(scalar_round_tripped.feedback_matrix_config);
    REQUIRE(scalar_options.rng_seed == 0x1234ABCDU);

    auto cascaded_config = MakeTimeVaryingFDNConfig();
    cascaded_config.feedback_matrix_config = sfFDN::CascadedFeedbackMatrixOptions{
        .matrix_size = cascaded_config.fdn_size,
        .stage_count = 2U,
        .sparsity = 2.5f,
        .type = sfFDN::ScalarMatrixType::Random,
        .gain_per_samples = 0.98f,
        .rng_seed = 0x5EED1234U,
    };
    const auto cascaded_round_tripped = nlohmann::json(cascaded_config).get<sfFDN::FDNConfig>();
    const auto& cascaded_options =
        std::get<sfFDN::CascadedFeedbackMatrixOptions>(cascaded_round_tripped.feedback_matrix_config);
    REQUIRE(cascaded_options.rng_seed == 0x5EED1234U);

    const auto original_fdn = sfFDN::CreateFDNFromConfig(cascaded_config);
    const auto round_tripped_fdn = sfFDN::CreateFDNFromConfig(cascaded_round_tripped);
    const auto original_output = RenderFDN(*original_fdn);
    const auto round_tripped_output = RenderFDN(*round_tripped_fdn);
    REQUIRE(std::ranges::any_of(original_output, [](float sample) { return sample != 0.f; }));
    for (const auto [actual, expected] : std::views::zip(round_tripped_output, original_output))
    {
        REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 2e-5f));
    }
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

TEST_CASE("JSON enum adapters reject unsupported representations", "[serialization]")
{
    const std::array invalid_values = {
        nlohmann::json("Unknown"),
        nlohmann::json(0),
        nlohmann::json(nullptr),
        nlohmann::json(true),
    };

    for (const auto& value : invalid_values)
    {
        REQUIRE_THROWS(value.get<sfFDN::ScalarMatrixType>());
        REQUIRE_THROWS(value.get<sfFDN::DelayInterpolationType>());
        REQUIRE_THROWS(value.get<sfFDN::DelayLengthType>());
        REQUIRE_THROWS(value.get<sfFDN::ParallelGainsMode>());
        REQUIRE_THROWS(value.get<sfFDN::TimeVaryingMatrixMode>());
    }

    REQUIRE(nlohmann::json("Count").get<sfFDN::ScalarMatrixType>() == sfFDN::ScalarMatrixType::Count);
    REQUIRE(nlohmann::json("Count").get<sfFDN::TimeVaryingMatrixMode>() == sfFDN::TimeVaryingMatrixMode::Count);

    REQUIRE_THROWS(nlohmann::json(static_cast<sfFDN::ScalarMatrixType>(255)));
    REQUIRE_THROWS(nlohmann::json(static_cast<sfFDN::DelayInterpolationType>(255)));
    REQUIRE_THROWS(nlohmann::json(static_cast<sfFDN::DelayLengthType>(255)));
    REQUIRE_THROWS(nlohmann::json(static_cast<sfFDN::ParallelGainsMode>(255)));
    REQUIRE_THROWS(nlohmann::json(static_cast<sfFDN::TimeVaryingMatrixMode>(255)));
}

TEST_CASE("JSON readers enforce numeric kinds and ranges", "[serialization]")
{
    const std::array invalid_sizes = {
        nlohmann::json(1.0),  nlohmann::json(1.5), nlohmann::json(-1),
        nlohmann::json(true), nlohmann::json("1"), nlohmann::json(std::numeric_limits<uint64_t>::max()),
    };
    const auto max_uint32 = nlohmann::json(std::numeric_limits<uint32_t>::max());

    const nlohmann::json scalar = {
        {"matrix_size", 0U},
        {"type", "Identity"},
        {"rng_seed", 0U},
    };
    const nlohmann::json cascaded = {
        {"matrix_size", 0U},  {"stage_count", 0U},       {"sparsity", 1.F},
        {"type", "Identity"}, {"gain_per_samples", 1.F}, {"rng_seed", 0U},
    };
    const nlohmann::json delay = {
        {"delay", 1.F},
        {"max_delay", 0U},
        {"interp_type", "None"},
    };
    const nlohmann::json delay_bank = {
        {"delays", nlohmann::json::array()},
        {"block_size", 0U},
        {"interpolation_type", "None"},
    };
    const nlohmann::json varying_delay_bank = {
        {"delays", nlohmann::json::array()},
        {"max_delay", 0U},
        {"interpolation_type", "None"},
        {"time_varying_config", nlohmann::json::array()},
    };
    const nlohmann::json varying_matrix = {
        {"matrix_size", 0U},
        {"mode", "Hadamard"},
        {"time_varying_config", nlohmann::json::array()},
        {"rng_seed", 0U},
    };
    REQUIRE_NOTHROW(scalar.get<sfFDN::ScalarFeedbackMatrixOptions>());
    REQUIRE_NOTHROW(cascaded.get<sfFDN::CascadedFeedbackMatrixOptions>());
    REQUIRE_NOTHROW(delay.get<sfFDN::DelayOptions>());
    REQUIRE_NOTHROW(delay_bank.get<sfFDN::DelayBankOptions>());
    REQUIRE_NOTHROW(varying_delay_bank.get<sfFDN::DelayBankTimeVaryingOptions>());
    REQUIRE_NOTHROW(varying_matrix.get<sfFDN::TimeVaryingFeedbackMatrixOptions>());

    for (const auto& size : invalid_sizes)
    {
        auto malformed = scalar;
        for (const auto* field : {"matrix_size", "rng_seed"})
        {
            malformed = scalar;
            malformed[field] = size;
            REQUIRE_THROWS(malformed.get<sfFDN::ScalarFeedbackMatrixOptions>());
        }

        for (const auto* field : {"matrix_size", "stage_count", "rng_seed"})
        {
            malformed = cascaded;
            malformed[field] = size;
            REQUIRE_THROWS(malformed.get<sfFDN::CascadedFeedbackMatrixOptions>());
        }

        malformed = delay;
        malformed["max_delay"] = size;
        REQUIRE_THROWS(malformed.get<sfFDN::DelayOptions>());

        malformed = delay_bank;
        malformed["block_size"] = size;
        REQUIRE_THROWS(malformed.get<sfFDN::DelayBankOptions>());

        malformed = varying_delay_bank;
        malformed["max_delay"] = size;
        REQUIRE_THROWS(malformed.get<sfFDN::DelayBankTimeVaryingOptions>());

        for (const auto* field : {"matrix_size", "rng_seed"})
        {
            malformed = varying_matrix;
            malformed[field] = size;
            REQUIRE_THROWS(malformed.get<sfFDN::TimeVaryingFeedbackMatrixOptions>());
        }

        for (const auto* field : {"fdn_size", "block_size"})
        {
            malformed = MakeTimeVaryingFDNConfig();
            malformed[field] = size;
            REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
        }
    }

    auto bounded_scalar = scalar;
    bounded_scalar["matrix_size"] = max_uint32;
    bounded_scalar["rng_seed"] = max_uint32;
    REQUIRE_NOTHROW(bounded_scalar.get<sfFDN::ScalarFeedbackMatrixOptions>());

    auto bounded_cascaded = cascaded;
    bounded_cascaded["matrix_size"] = max_uint32;
    bounded_cascaded["stage_count"] = max_uint32;
    bounded_cascaded["rng_seed"] = max_uint32;
    REQUIRE_NOTHROW(bounded_cascaded.get<sfFDN::CascadedFeedbackMatrixOptions>());

    auto bounded_delay = delay;
    bounded_delay["max_delay"] = max_uint32;
    REQUIRE_NOTHROW(bounded_delay.get<sfFDN::DelayOptions>());

    auto bounded_delay_bank = delay_bank;
    bounded_delay_bank["block_size"] = max_uint32;
    REQUIRE_NOTHROW(bounded_delay_bank.get<sfFDN::DelayBankOptions>());

    auto bounded_varying_delay_bank = varying_delay_bank;
    bounded_varying_delay_bank["max_delay"] = max_uint32;
    REQUIRE_NOTHROW(bounded_varying_delay_bank.get<sfFDN::DelayBankTimeVaryingOptions>());

    auto bounded_varying_matrix = varying_matrix;
    bounded_varying_matrix["matrix_size"] = max_uint32;
    bounded_varying_matrix["rng_seed"] = max_uint32;
    REQUIRE_NOTHROW(bounded_varying_matrix.get<sfFDN::TimeVaryingFeedbackMatrixOptions>());

    auto bounded_root = nlohmann::json(MakeTimeVaryingFDNConfig());
    bounded_root["fdn_size"] = max_uint32;
    bounded_root["block_size"] = max_uint32;
    REQUIRE_NOTHROW(bounded_root.get<sfFDN::FDNConfig>());

    const nlohmann::json sparse_fir = sfFDN::SparseFirOptions{.coeffs = {{0U, 1.F}}};
    REQUIRE(sparse_fir["coeffs"] == nlohmann::json::array({nlohmann::json::array({0U, 1.F})}));
    REQUIRE_NOTHROW(sparse_fir.get<sfFDN::SparseFirOptions>());
    for (const auto& size : invalid_sizes)
    {
        auto malformed = sparse_fir;
        malformed["coeffs"][0][0] = size;
        REQUIRE_THROWS(malformed.get<sfFDN::SparseFirOptions>());
    }

    auto bounded_sparse_fir = sparse_fir;
    bounded_sparse_fir["coeffs"][0][0] = max_uint32;
    REQUIRE_NOTHROW(bounded_sparse_fir.get<sfFDN::SparseFirOptions>());

    auto fractional_rate = MakeTimeVaryingFDNConfig();
    nlohmann::json fractional_rate_json = fractional_rate;
    fractional_rate_json["sample_rate"] = 48000.5;
    REQUIRE_THAT(fractional_rate_json.get<sfFDN::FDNConfig>().sample_rate, Catch::Matchers::WithinAbs(48000.5F, 0.F));

    for (const auto& value :
         {nlohmann::json("48000"), nlohmann::json(true), nlohmann::json(std::numeric_limits<double>::infinity()),
          nlohmann::json(std::numeric_limits<double>::max())})
    {
        auto malformed = fractional_rate_json;
        malformed["sample_rate"] = value;
        REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
    }
}

TEST_CASE("JSON readers preserve optional and transactional destinations", "[serialization]")
{
    const sfFDN::ScalarFeedbackMatrixOptions populated_matrix = {
        .matrix_size = 2U,
        .type = sfFDN::ScalarMatrixType::VariableDiffusion,
        .custom_matrix = std::vector<float>{1.F, 0.F, 0.F, 1.F},
        .rng_seed = 7U,
        .arg = 0.5F,
    };
    auto absent_optionals = nlohmann::json(populated_matrix);
    absent_optionals.erase("custom_matrix");
    absent_optionals.erase("arg");
    auto cleared_matrix = populated_matrix;
    absent_optionals.get_to(cleared_matrix);
    REQUIRE_FALSE(cleared_matrix.custom_matrix.has_value());
    REQUIRE_FALSE(cleared_matrix.arg.has_value());

    auto null_optionals = nlohmann::json(populated_matrix);
    null_optionals["custom_matrix"] = nullptr;
    null_optionals["arg"] = nullptr;
    null_optionals.get_to(cleared_matrix);
    REQUIRE_FALSE(cleared_matrix.custom_matrix.has_value());
    REQUIRE_FALSE(cleared_matrix.arg.has_value());

    auto empty_matrix = nlohmann::json(populated_matrix);
    empty_matrix["custom_matrix"] = nlohmann::json::array();
    empty_matrix.get_to(cleared_matrix);
    REQUIRE(cleared_matrix.custom_matrix == std::vector<float>{});

    const sfFDN::DelayOptions populated_delay = {
        .delay = 2.F,
        .max_delay = 4U,
        .interp_type = sfFDN::DelayInterpolationType::Linear,
        .lfo_config = sfFDN::ModulationOptions{.frequency = 0.01F, .amplitude = 0.25F, .initial_phase = 0.F},
    };
    auto absent_lfo = nlohmann::json(populated_delay);
    absent_lfo.erase("lfo_config");
    auto cleared_delay = populated_delay;
    absent_lfo.get_to(cleared_delay);
    REQUIRE_FALSE(cleared_delay.lfo_config.has_value());
    auto null_lfo = nlohmann::json(populated_delay);
    null_lfo["lfo_config"] = nullptr;
    null_lfo.get_to(cleared_delay);
    REQUIRE_FALSE(cleared_delay.lfo_config.has_value());

    auto malformed_matrix = nlohmann::json(populated_matrix);
    malformed_matrix["arg"] = "late";
    RequireUnchangedAfterFailedRead(malformed_matrix, populated_matrix);

    const sfFDN::ParallelGainsOptions populated_gains = {
        .mode = sfFDN::ParallelGainsMode::Parallel,
        .gains = {0.5F},
        .time_varying_config = {},
    };
    auto malformed_gains = nlohmann::json(populated_gains);
    malformed_gains["time_varying_config"] = nlohmann::json::object();
    RequireUnchangedAfterFailedRead(malformed_gains, populated_gains);

    const sfFDN::MultichannelProcessorOptions populated_bank = {
        .channels = {sfFDN::FirOptions{.coeffs = {1.F}}, sfFDN::AllpassFilterOptions{.coeff = 0.5F}},
    };
    auto malformed_bank = nlohmann::json(populated_bank);
    malformed_bank["channels"][1] = nlohmann::json::object();
    RequireUnchangedAfterFailedRead(malformed_bank, populated_bank);

    const sfFDN::AttenuationFilterBankOptions populated_attenuation = {
        .filter_configs = {sfFDN::HomogenousFilterOptions{}, sfFDN::TwoBandFilterOptions{}},
    };
    auto malformed_attenuation = nlohmann::json(populated_attenuation).at("AttenuationFilterBankOptions");
    malformed_attenuation[1]["TwoBandFilterConfig"]["t60s"] = nlohmann::json::array({1.F});
    RequireUnchangedAfterFailedRead(malformed_attenuation, populated_attenuation);

    auto cleared_attenuation = populated_attenuation;
    nlohmann::json::array().get_to(cleared_attenuation);
    REQUIRE(cleared_attenuation.filter_configs.empty());

    auto malformed_root = nlohmann::json(MakeTimeVaryingFDNConfig());
    malformed_root["input_block_config"]["single_channel_processors"] =
        nlohmann::json::array({{{"FirOptions", {{"coeffs", {1.F}}}}}});
    malformed_root["tone_correction_filters"] = nlohmann::json::object();
    RequireUnchangedAfterFailedRead(malformed_root, MakeTimeVaryingFDNConfig());
}

TEST_CASE("JSON readers require exact array and wrapper forms", "[serialization]")
{
    const nlohmann::json valid = MakeTimeVaryingFDNConfig();

    for (const auto& shape : {nlohmann::json::object(), nlohmann::json(nullptr)})
    {
        auto malformed = valid;
        malformed["input_block_config"]["single_channel_processors"] = shape;
        REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());

        malformed = valid;
        malformed["loop_filter_configs"] = shape;
        REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
    }

    REQUIRE_NOTHROW(nlohmann::json{{"coeffs", nlohmann::json::array()}}.get<sfFDN::FirOptions>());
    REQUIRE_NOTHROW(nlohmann::json{{"channels", nlohmann::json::array()}}.get<sfFDN::MultichannelProcessorOptions>());

    const nlohmann::json two_band = {
        {"t60s", {1.F, 0.5F}},
        {"delay", 1.F},
        {"sample_rate", 48000.F},
    };
    for (const auto& t60s : {nlohmann::json::array({1.F}), nlohmann::json::array({1.F, 0.5F, 0.25F}),
                             nlohmann::json::object(), nlohmann::json(nullptr)})
    {
        auto malformed = two_band;
        malformed["t60s"] = t60s;
        REQUIRE_THROWS(malformed.get<sfFDN::TwoBandFilterOptions>());
    }

    const nlohmann::json sparse_fir = sfFDN::SparseFirOptions{.coeffs = {{0U, 1.F}}};
    REQUIRE(sparse_fir["coeffs"] == nlohmann::json::array({nlohmann::json::array({0U, 1.F})}));
    for (const auto& pair : {nlohmann::json::array({0U}), nlohmann::json::array({0U, 1.F, 2.F}),
                             nlohmann::json::object(), nlohmann::json(nullptr)})
    {
        auto malformed = sparse_fir;
        malformed["coeffs"][0] = pair;
        REQUIRE_THROWS(malformed.get<sfFDN::SparseFirOptions>());
    }

    const nlohmann::json single = {{"FirOptions", {{"coeffs", {1.F}}}}};
    const nlohmann::json multi = {
        {"ParallelGainsConfig",
         {{"mode", "Parallel"}, {"gains", {1.F}}, {"time_varying_config", nlohmann::json::array()}}},
    };
    const nlohmann::json matrix = {
        {"ScalarFeedbackMatrixOptions", {{"matrix_size", 1U}, {"type", "Identity"}, {"rng_seed", 0U}}},
    };

    for (const auto& wrapper :
         {nlohmann::json{{"FirOptions", {{"coeffs", {1.F}}}},
                         {"DelayOptions", {{"delay", 1.F}, {"max_delay", 1U}, {"interp_type", "None"}}}},
          nlohmann::json{{"FirOptions", {{"coeffs", {1.F}}}}, {"UnknownOptions", {}}},
          nlohmann::json{{"metadata", nlohmann::json::object()}},
          nlohmann::json{{"FirOptions", {{"coeffs", {1.F}}}}, {"metadata", nlohmann::json::object()}},
          nlohmann::json{{"FirOptions", {{"coeffs", {1.F}}}}, {"metadata", "preset"}}})
    {
        REQUIRE_THROWS(sfFDN::SingleChannelProcessorFromJson(wrapper));
    }

    for (const auto& wrapper :
         {nlohmann::json{{"ParallelGainsConfig", multi.at("ParallelGainsConfig")},
                         {"DelayBankOptions", {{"delays", {}}, {"block_size", 1U}, {"interpolation_type", "None"}}}},
          nlohmann::json{{"ParallelGainsConfig", multi.at("ParallelGainsConfig")}, {"UnknownOptions", {}}},
          nlohmann::json{{"metadata", nlohmann::json::object()}},
          nlohmann::json{{"ParallelGainsConfig", multi.at("ParallelGainsConfig")},
                         {"metadata", nlohmann::json::object()}},
          nlohmann::json{{"ParallelGainsConfig", multi.at("ParallelGainsConfig")}, {"metadata", "preset"}}})
    {
        REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(wrapper));
    }

    for (const auto& wrapper :
         {nlohmann::json{{"ScalarFeedbackMatrixOptions", matrix.at("ScalarFeedbackMatrixOptions")},
                         {"CascadedFeedbackMatrixInfo",
                          {{"matrix_size", 1U},
                           {"stage_count", 1U},
                           {"sparsity", 1.F},
                           {"type", "Identity"},
                           {"gain_per_samples", 1.F},
                           {"rng_seed", 0U}}}},
          nlohmann::json{{"ScalarFeedbackMatrixOptions", matrix.at("ScalarFeedbackMatrixOptions")},
                         {"UnknownOptions", {}}},
          nlohmann::json{{"metadata", nlohmann::json::object()}},
          nlohmann::json{{"ScalarFeedbackMatrixOptions", matrix.at("ScalarFeedbackMatrixOptions")},
                         {"metadata", nlohmann::json::object()}},
          nlohmann::json{{"ScalarFeedbackMatrixOptions", matrix.at("ScalarFeedbackMatrixOptions")},
                         {"metadata", "preset"}}})
    {
        auto malformed = valid;
        malformed["feedback_matrix_config"] = wrapper;
        REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
    }

    const nlohmann::json attenuation = {
        {"AttenuationFilterBankOptions",
         {{{"TwoBandFilterConfig", {{"t60s", {1.F, 0.5F}}, {"delay", 1.F}, {"sample_rate", 48000.F}}}}}},
    };
    auto root_attenuation = valid;
    root_attenuation["attenuation_filter_bank_config"] = attenuation;
    const auto parsed_root = root_attenuation.get<sfFDN::FDNConfig>();
    REQUIRE(parsed_root.attenuation_filter_bank_config.has_value());
    REQUIRE(parsed_root.attenuation_filter_bank_config->filter_configs.size() == 1U);
    REQUIRE(std::holds_alternative<sfFDN::TwoBandFilterOptions>(
        parsed_root.attenuation_filter_bank_config->filter_configs.front()));

    for (const auto& wrapper :
         {nlohmann::json{{"AttenuationFilterBankOptions", attenuation.at("AttenuationFilterBankOptions")},
                         {"UnknownKey", nlohmann::json::object()}},
          nlohmann::json{{"metadata", nlohmann::json::object()}},
          nlohmann::json{{"AttenuationFilterBankOptions", attenuation.at("AttenuationFilterBankOptions")},
                         {"metadata", nlohmann::json::object()}},
          nlohmann::json{{"AttenuationFilterBankOptions", attenuation.at("AttenuationFilterBankOptions")},
                         {"metadata", "preset"}},
          nlohmann::json{{"AttenuationFilterBankOptions", attenuation.at("AttenuationFilterBankOptions")},
                         {"metadata", nlohmann::json::object()},
                         {"UnknownKey", nlohmann::json::object()}}})
    {
        auto malformed = valid;
        malformed["attenuation_filter_bank_config"] = wrapper;
        REQUIRE_THROWS(malformed.get<sfFDN::FDNConfig>());
    }

    auto metadata_attenuation = attenuation;
    metadata_attenuation["metadata"] = nlohmann::json::object();
    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(metadata_attenuation));
    auto ambiguous_filter = attenuation;
    ambiguous_filter["AttenuationFilterBankOptions"][0]["UnknownFilter"] = nlohmann::json::object();
    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(ambiguous_filter));
    auto ambiguous_attenuation = attenuation;
    ambiguous_attenuation["DelayBankOptions"] = {{"delays", {}}, {"block_size", 1U}, {"interpolation_type", "None"}};
    REQUIRE_THROWS(sfFDN::MultichannelProcessorFromJson(ambiguous_attenuation));
}

TEST_CASE("Stage gains and named stages use their dedicated JSON contracts", "[serialization]")
{
    const sfFDN::StageGainsOptions gains = {
        .gains = {0.25F, 0.5F},
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.25F, .initial_phase = 0.5F},
                                {.frequency = 0.002F, .amplitude = -0.25F, .initial_phase = 0.25F}},
    };
    const nlohmann::json gains_json = gains;
    REQUIRE(gains_json.size() == 2U);
    REQUIRE(gains_json.contains("gains"));
    REQUIRE(gains_json.contains("time_varying_config"));
    REQUIRE_FALSE(gains_json.contains("mode"));
    REQUIRE(gains_json.get<sfFDN::StageGainsOptions>() == gains);

    nlohmann::json reused_gains = {
        {"mode", "Split"},
        {"stale", true},
    };
    sfFDN::to_json(reused_gains, gains);
    REQUIRE(reused_gains == gains_json);
    REQUIRE(reused_gains.get<sfFDN::StageGainsOptions>() == gains);

    auto obsolete_stage_gains = gains_json;
    obsolete_stage_gains["mode"] = "Split";
    REQUIRE_THROWS(obsolete_stage_gains.get<sfFDN::StageGainsOptions>());

    const sfFDN::InputStageConfig input = {
        .single_channel_processors = {sfFDN::AllpassFilterOptions{.coeff = 0.25F}},
        .parallel_gains_config = gains,
        .multichannel_processors = {sfFDN::ParallelGainsOptions{
            .mode = sfFDN::ParallelGainsMode::Parallel, .gains = {1.F, 1.F}, .time_varying_config = {}}},
    };
    const sfFDN::OutputStageConfig output = {
        .multichannel_processors = {sfFDN::ParallelGainsOptions{
            .mode = sfFDN::ParallelGainsMode::Parallel, .gains = {1.F, 1.F}, .time_varying_config = {}}},
        .parallel_gains_config = gains,
        .single_channel_processors = {sfFDN::FirOptions{.coeffs = {1.F}}},
    };
    const nlohmann::json input_json = input;
    const nlohmann::json output_json = output;
    REQUIRE(input_json.get<sfFDN::InputStageConfig>() == input);
    REQUIRE(output_json.get<sfFDN::OutputStageConfig>() == output);
    REQUIRE(input_json["parallel_gains_config"] == gains_json);
    REQUIRE(output_json["parallel_gains_config"] == gains_json);

    auto malformed_input = input_json;
    malformed_input["parallel_gains_config"]["time_varying_config"] = nlohmann::json::object();
    auto retained_input = input;
    REQUIRE_THROWS(malformed_input.get_to(retained_input));
    REQUIRE(retained_input == input);

    auto malformed_output = output_json;
    malformed_output["parallel_gains_config"]["mode"] = "Merge";
    auto retained_output = output;
    REQUIRE_THROWS(malformed_output.get_to(retained_output));
    REQUIRE(retained_output == output);

    const sfFDN::ParallelGainsOptions parallel = {
        .mode = sfFDN::ParallelGainsMode::Parallel,
        .gains = {0.25F, 0.5F},
        .time_varying_config = {},
    };
    const nlohmann::json parallel_json = parallel;
    REQUIRE(parallel_json.size() == 3U);
    REQUIRE(parallel_json.contains("mode"));
    REQUIRE(parallel_json.get<sfFDN::ParallelGainsOptions>() == parallel);
}

TEST_CASE("FDNConfig JSON round-trips and fails transactionally by structural equality", "[serialization]")
{
    const auto original = MakeTimeVaryingFDNConfig();
    const nlohmann::json serialized = original;
    REQUIRE(serialized.get<sfFDN::FDNConfig>() == original);

    auto malformed = serialized;
    malformed["output_block_config"]["parallel_gains_config"]["gains"] = nlohmann::json::object();
    auto retained = original;
    REQUIRE_THROWS(malformed.get_to(retained));
    REQUIRE(retained == original);
}
