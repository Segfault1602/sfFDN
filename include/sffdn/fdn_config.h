#pragma once

#include "sffdn/fdn.h"
#include "sffdn/types.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace sfFDN
{

/** @brief Configuration for the FDN.
 */
struct FDNConfig
{
    //! Size of the FDN (number of channels)
    uint32_t fdn_size;

    //! Whether to use transposed configuration
    bool transposed;

    //! Direct path gain
    float direct_gain;

    //! Internal block size for processing audio. Ideally should match the block size of the system.
    uint32_t block_size;

    //! Sample rate for the FDN. This is used to configure time-based components like delays and filters.
    float sample_rate;

    //! Delay bank configuration
    DelayBankOptions delay_bank_config;

    //! Input gain Block
    struct
    {
        //! A vector of single-channel processors to apply to the input signal before it gets split into multiple
        //! channels.
        std::vector<single_channel_processor_variant_t> single_channel_processors;
        //! Configuration for parallel gain processing applied to the input signal. Must always be in Split mode.
        ParallelGainsOptions parallel_gains_config{.mode = ParallelGainsMode::Split, .gains = {}};
        //! A vector of multi-channel processors to apply to the input signal after the parallel gains.
        std::vector<multi_channel_processor_variant_t> multichannel_processors;
    } input_block_config;

    //! Feedback matrix block
    feedback_matrix_variant_t feedback_matrix_config;

    //! Loop filter block
    std::vector<multi_channel_processor_variant_t> loop_filter_configs;

    //! Output gain block
    struct
    {
        //! A vector of multi-channel processors to apply to the output signal before it gets mixed down to a single
        //! channel.
        std::vector<multi_channel_processor_variant_t> multichannel_processors;
        //! Configuration for parallel gain processing applied to the output signal. Must always be in Merge mode.
        ParallelGainsOptions parallel_gains_config{.mode = ParallelGainsMode::Merge, .gains = {}};
        //! A vector of single-channel processors to apply to the output signal after it gets mixed down to a single
        //! channel.
        std::vector<single_channel_processor_variant_t> single_channel_processors;
    } output_block_config;

    //! Tone correction filter block
    std::vector<single_channel_processor_variant_t> tone_correction_filters;
};

std::unique_ptr<FDN> CreateFDNFromConfig(const FDNConfig& config);

void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p);
void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p);

} // namespace sfFDN