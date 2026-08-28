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

    nlohmann::json j = config;

    std::cout << j.dump(4) << std::endl;

    sfFDN::FDNConfig deserialized_config = j.get<sfFDN::FDNConfig>();

    const auto& single_channel_procs = deserialized_config.input_block_config.single_channel_processors;
    REQUIRE(single_channel_procs.size() == 3);
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
