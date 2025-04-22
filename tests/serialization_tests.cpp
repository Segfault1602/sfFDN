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

    config.input_block_config.single_channel_processors = {sfFDN::AllpassFilterOptions{.coeff = 0.5f},
                                                           sfFDN::DelayOptions{.delay = 64}};
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
    config.loop_filter_configs.push_back(attenuation_filter_bank_config);

    config.output_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Merge, {0.7f, 0.6f, 0.5f, 0.4f}, {}};

    nlohmann::json j = config;

    std::cout << j.dump(4) << std::endl;

    sfFDN::FDNConfig deserialized_config = j.get<sfFDN::FDNConfig>();
}