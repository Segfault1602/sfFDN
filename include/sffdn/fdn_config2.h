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

using multichannel_processor_variant_t =
    std::variant<ParallelGainsOptions, ParallelSchroederAllpassSectionOptions, AttenuationFilterBankOptions,
                 DelayBankOptions, DelayBankTimeVaryingOptions, CascadedFeedbackMatrixOptions,
                 ScalarFeedbackMatrixOptions>;

struct FDNConfig2
{
    uint32_t fdn_size; // number of channels
    bool transposed;
    float direct_gain;
    uint32_t block_size;
    uint32_t sample_rate;
    DelayBankOptions delay_bank_config;

    // Input gain Block
    struct
    {
        std::vector<single_channel_processor_variant_t> single_channel_processors;
        ParallelGainsOptions parallel_gains_config;
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
        ParallelGainsOptions parallel_gains_config;
        std::vector<single_channel_processor_variant_t> single_channel_processors;
    } output_block_config;

    // Tone correction filter block
    std::vector<single_channel_processor_variant_t> tone_correction_filters;
};

std::unique_ptr<FDN> CreateFDNFromConfig2(const FDNConfig2& config);

void to_json(nlohmann::json& j, const sfFDN::FDNConfig2& p);
void from_json(const nlohmann::json& j, sfFDN::FDNConfig2& p);

} // namespace sfFDN