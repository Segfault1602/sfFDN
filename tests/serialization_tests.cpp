#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

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

} // namespace

TEST_CASE("FDNConfig")
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

    sfFDN::MultichannelDattorroDelayOptions dattorro_bank_config;
    dattorro_bank_config.delays.resize(4);
    for (size_t i = 0; i < dattorro_bank_config.delays.size(); ++i)
    {
        auto& channel = dattorro_bank_config.delays[i];
        channel.blend = 0.5f + (0.01f * static_cast<float>(i));
        channel.feedforward = 1.f;
        channel.feedback = 0.25f;
        channel.delay_config.delay = 64.f + (8.f * static_cast<float>(i));
        channel.delay_config.max_delay = 256;
        channel.delay_config.interp_type = sfFDN::DelayInterpolationType::Allpass;
        // Leave the last channel unmodulated, so that the optional lfo_config is exercised both ways.
        if (i + 1 < dattorro_bank_config.delays.size())
        {
            channel.delay_config.lfo_config = sfFDN::ModulationOptions{
                .frequency = 0.0002f, .amplitude = 4.f, .initial_phase = 0.25f * static_cast<float>(i)};
        }
        else
        {
            channel.delay_config.lfo_config = std::nullopt;
        }
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

    sfFDN::MultichannelControllableFullWaveRectifierOptions rectifier_bank_config;
    rectifier_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < rectifier_bank_config.channels.size(); ++i)
    {
        rectifier_bank_config.channels[i] =
            sfFDN::ControllableFullWaveRectifierOptions{.alpha = 0.1f * static_cast<float>(i + 1),
                                                        .antialiasing = (i % 2) == 0,
                                                        .dc_block = (i % 2) == 1,
                                                        .sample_rate = 48000.f};
    }

    sfFDN::MultichannelSignalDependentFractionalDelayOptions sdfd_bank_config;
    sdfd_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < sdfd_bank_config.channels.size(); ++i)
    {
        sdfd_bank_config.channels[i] =
            sfFDN::SignalDependentFractionalDelayOptions{.d = 0.2f * static_cast<float>(i + 1)};
    }

    sfFDN::MultichannelRingModulatorOptions ring_mod_bank_config;
    ring_mod_bank_config.channels.resize(4);
    for (size_t i = 0; i + 1 < ring_mod_bank_config.channels.size(); ++i)
    {
        ring_mod_bank_config.channels[i] =
            sfFDN::RingModulatorOptions{.frequency = 0.001f * static_cast<float>(i + 1),
                                        .amplitude = 1.4142f,
                                        .initial_phase = 0.25f * static_cast<float>(i)};
    }

    config.loop_filter_configs = {rectifier_bank_config, sdfd_bank_config, ring_mod_bank_config};

    nlohmann::json j = config;

    std::cout << j.dump(4) << std::endl;

    sfFDN::FDNConfig deserialized_config = j.get<sfFDN::FDNConfig>();

    const auto& single_channel_procs = deserialized_config.input_block_config.single_channel_processors;
    REQUIRE(single_channel_procs.size() == 6);
    REQUIRE(std::holds_alternative<sfFDN::DattorroDelayOptions>(single_channel_procs[2]));

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
    REQUIRE(std::holds_alternative<sfFDN::MultichannelDattorroDelayOptions>(multichannel_procs[0]));

    const auto& dattorro_bank = std::get<sfFDN::MultichannelDattorroDelayOptions>(multichannel_procs[0]);
    REQUIRE(dattorro_bank.delays.size() == 4);
    for (size_t i = 0; i < dattorro_bank.delays.size(); ++i)
    {
        const auto& channel = dattorro_bank.delays[i];
        REQUIRE_THAT(channel.blend, Catch::Matchers::WithinAbs(0.5f + (0.01f * static_cast<float>(i)), 1e-5f));
        REQUIRE_THAT(channel.feedforward, Catch::Matchers::WithinAbs(1.f, 1e-5f));
        REQUIRE_THAT(channel.feedback, Catch::Matchers::WithinAbs(0.25f, 1e-5f));
        REQUIRE_THAT(channel.delay_config.delay,
                     Catch::Matchers::WithinAbs(64.f + (8.f * static_cast<float>(i)), 1e-5f));
        REQUIRE(channel.delay_config.interp_type == sfFDN::DelayInterpolationType::Allpass);

        if (i + 1 < dattorro_bank.delays.size())
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

    REQUIRE(std::holds_alternative<sfFDN::MultichannelControllableFullWaveRectifierOptions>(loop_filters[0]));
    const auto& rectifier_bank = std::get<sfFDN::MultichannelControllableFullWaveRectifierOptions>(loop_filters[0]);
    REQUIRE(rectifier_bank.channels.size() == 4);
    REQUIRE_FALSE(rectifier_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < rectifier_bank.channels.size(); ++i)
    {
        REQUIRE(rectifier_bank.channels[i].has_value());
        REQUIRE_THAT(rectifier_bank.channels[i]->alpha,
                     Catch::Matchers::WithinAbs(0.1f * static_cast<float>(i + 1), 1e-6f));
        REQUIRE(rectifier_bank.channels[i]->antialiasing == ((i % 2) == 0));
        REQUIRE(rectifier_bank.channels[i]->dc_block == ((i % 2) == 1));
    }

    REQUIRE(std::holds_alternative<sfFDN::MultichannelSignalDependentFractionalDelayOptions>(loop_filters[1]));
    const auto& sdfd_bank = std::get<sfFDN::MultichannelSignalDependentFractionalDelayOptions>(loop_filters[1]);
    REQUIRE(sdfd_bank.channels.size() == 4);
    REQUIRE_FALSE(sdfd_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < sdfd_bank.channels.size(); ++i)
    {
        REQUIRE(sdfd_bank.channels[i].has_value());
        REQUIRE_THAT(sdfd_bank.channels[i]->d, Catch::Matchers::WithinAbs(0.2f * static_cast<float>(i + 1), 1e-6f));
    }

    REQUIRE(std::holds_alternative<sfFDN::MultichannelRingModulatorOptions>(loop_filters[2]));
    const auto& ring_mod_bank = std::get<sfFDN::MultichannelRingModulatorOptions>(loop_filters[2]);
    REQUIRE(ring_mod_bank.channels.size() == 4);
    REQUIRE_FALSE(ring_mod_bank.channels[3].has_value());
    for (size_t i = 0; i + 1 < ring_mod_bank.channels.size(); ++i)
    {
        REQUIRE(ring_mod_bank.channels[i].has_value());
        REQUIRE_THAT(ring_mod_bank.channels[i]->frequency,
                     Catch::Matchers::WithinAbs(0.001f * static_cast<float>(i + 1), 1e-9f));
        REQUIRE_THAT(ring_mod_bank.channels[i]->initial_phase,
                     Catch::Matchers::WithinAbs(0.25f * static_cast<float>(i), 1e-6f));
    }

    // The configuration must also survive being turned into an actual FDN.
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(deserialized_config));
}

TEST_CASE("Time-varying feedback matrix options serialize")
{
    const auto options = MakeTimeVaryingMatrixOptions(4);

    nlohmann::json json = options;
    const auto deserialized_options = json.get<sfFDN::TimeVaryingFeedbackMatrixOptions>();

    RequireEqual(deserialized_options, options);
}

TEST_CASE("RealSchur time-varying feedback matrix JSON round-trip is reproducible")
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

TEST_CASE("FDNConfig serializes a time-varying feedback matrix")
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

TEST_CASE("FDNConfig creates an FDN with a time-varying feedback matrix")
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

TEST_CASE("FDNConfig rejects invalid time-varying feedback matrix sizes")
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

TEST_CASE("FDNConfig rejects invalid time-varying feedback matrix options before construction")
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

TEST_CASE("Time-varying Schroeder allpass options serialize")
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
    const sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions bank{
        .sections = {section, section},
    };

    const nlohmann::json section_json = section;
    const nlohmann::json bank_json = bank;
    RequireEqual(section_json.get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(), section);

    const auto round_tripped_bank = bank_json.get<sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions>();
    REQUIRE(round_tripped_bank.sections.size() == bank.sections.size());
    for (size_t index = 0; index < bank.sections.size(); ++index)
    {
        RequireEqual(round_tripped_bank.sections[index], bank.sections[index]);
    }
}

TEST_CASE("FDNConfig serializes and creates time-varying Schroeder allpasses")
{
    auto config = MakeTimeVaryingFDNConfig();
    const sfFDN::TimeVaryingSchroederAllpassSectionOptions input_section{
        .delays = {5.F},
        .gains = {0.4F},
        .time_varying_config = {{.frequency = 0.001F, .amplitude = 0.2F, .initial_phase = 0.25F}},
    };
    config.input_block_config.single_channel_processors.emplace_back(input_section);

    sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions loop_bank;
    for (uint32_t channel = 0; channel < config.fdn_size; ++channel)
    {
        loop_bank.sections.push_back({
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
    REQUIRE(json["loop_filter_configs"][0].contains("MultichannelTimeVaryingSchroederAllpassSectionOptions"));

    const auto round_tripped = json.get<sfFDN::FDNConfig>();
    REQUIRE(std::holds_alternative<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(
        round_tripped.input_block_config.single_channel_processors[0]));
    REQUIRE(std::holds_alternative<sfFDN::MultichannelTimeVaryingSchroederAllpassSectionOptions>(
        round_tripped.loop_filter_configs[0]));
    REQUIRE_NOTHROW(sfFDN::CreateFDNFromConfig(round_tripped));

    auto invalid = round_tripped;
    auto& invalid_section = std::get<sfFDN::TimeVaryingSchroederAllpassSectionOptions>(
        invalid.input_block_config.single_channel_processors[0]);
    invalid_section.gains[0] = 1.F;
    REQUIRE_THROWS_AS(sfFDN::CreateFDNFromConfig(invalid), std::runtime_error);
}
