#pragma once

#include "sffdn/delay_time_varying.h"
#include "sffdn/delaybank.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/filter.h"
#include "sffdn/filter_feedback_matrix.h"
#include "sffdn/parallel_gains.h"
#include "sffdn/schroeder_allpass.h"
#include "sffdn/sffdn.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace sfFDN
{

using feedback_matrix_variant_t =
    std::variant<CascadedFeedbackMatrixInfo, ScalarFeedbackMatrixConfig, std::vector<float>>;

using multichannel_processor_variant_t =
    std::variant<ParallelGainsConfig, ParallelSchroederAllpassSectionConfig, AttenuationFilterBankConfig,
                 DelayBankConfig, DelayBankTimeVaryingConfig, CascadedFeedbackMatrixInfo, ScalarFeedbackMatrixConfig>;

using single_channel_processor_variant_t =
    std::variant<SchroederAllpassSectionConfig, AllpassFilterConfig, CascadedBiquadsConfig, FirConfig, DelayConfig>;

struct FDNConfig2
{
    uint32_t fdn_size; // number of channels
    bool transposed;
    float direct_gain;
    uint32_t block_size;
    uint32_t sample_rate;
    DelayBankConfig delay_bank_config;

    // Input gain Block
    struct
    {
        std::vector<single_channel_processor_variant_t> single_channel_processors;
        ParallelGainsConfig parallel_gains_config;
        std::vector<multichannel_processor_variant_t> multichannel_processors;
    } input_block_config;

    // Feedback matrix block
    feedback_matrix_variant_t feedback_matrix_config;

    // Loop filter block
    std::vector<multichannel_processor_variant_t> loop_filter_configs;

    // Output gain block
    struct
    {
        std::vector<multichannel_processor_variant_t> multichannel_processors;
        ParallelGainsConfig parallel_gains_config;
        std::vector<single_channel_processor_variant_t> single_channel_processors;
    } output_block_config;

    // Tone correction filter block
    std::vector<single_channel_processor_variant_t> tone_correction_filters;
};

std::unique_ptr<FDN> CreateFDNFromConfig2(const FDNConfig2& config);
} // namespace sfFDN