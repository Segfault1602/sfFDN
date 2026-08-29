#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"
#include "test_utils.h"

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
