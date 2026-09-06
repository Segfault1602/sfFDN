#pragma once

#include "sffdn/config_diagnostics.h"
#include "sffdn/fdn.h"
#include "sffdn/types.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace sfFDN
{

/** @brief Configuration for the FDN.
 */
struct FDNConfig
{
    //! Size of the FDN (number of channels)
    uint32_t fdn_size{0};

    //! Whether to use transposed configuration
    bool transposed{false};

    //! Direct path gain
    float direct_gain{0.f};

    //! Internal block size for processing audio. Ideally should match the block size of the system.
    uint32_t block_size{kDefaultBlockSize};

    //! Sample rate for the FDN. This is used to configure time-based components like delays and filters.
    float sample_rate{static_cast<float>(kDefaultSampleRate)};

    //! Delay bank configuration. Its block size must be nonzero and at least this configuration's block size.
    DelayBankOptions delay_bank_config;

    //! Input gain Block
    struct
    {
        //! A vector of single-channel processors to apply to the input signal before it gets split into multiple
        //! channels.
        std::vector<single_channel_processor_variant_t> single_channel_processors;
        //! Configuration for parallel gain processing applied to the input signal. Must always be in Split mode.
        ParallelGainsOptions parallel_gains_config{
            .mode = ParallelGainsMode::Split, .gains = {}, .time_varying_config = {}};
        //! A vector of multi-channel processors to apply to the input signal after the parallel gains. A
        //! MultichannelProcessorOptions bank has exactly fdn_size channels.
        std::vector<multi_channel_processor_variant_t> multichannel_processors;
    } input_block_config;

    //! Feedback matrix block
    feedback_matrix_variant_t feedback_matrix_config;

    //! Attenuation filter bank block
    std::optional<AttenuationFilterBankOptions> attenuation_filter_bank_config;

    //! Loop filter block. A MultichannelProcessorOptions bank has exactly fdn_size channels.
    std::vector<multi_channel_processor_variant_t> loop_filter_configs;

    //! Output gain block
    struct
    {
        //! A vector of multi-channel processors to apply to the output signal before it gets mixed down to a single
        //! channel. A MultichannelProcessorOptions bank has exactly fdn_size channels.
        std::vector<multi_channel_processor_variant_t> multichannel_processors;
        //! Configuration for parallel gain processing applied to the output signal. Must always be in Merge mode.
        ParallelGainsOptions parallel_gains_config{
            .mode = ParallelGainsMode::Merge, .gains = {}, .time_varying_config = {}};
        //! A vector of single-channel processors to apply to the output signal after it gets mixed down to a single
        //! channel.
        std::vector<single_channel_processor_variant_t> single_channel_processors;
    } output_block_config;

    //! Tone correction filter block
    std::vector<single_channel_processor_variant_t> tone_correction_filters;
};

/** @brief Validates FDN graph structure, dimensions, and delay, gain, and allpass option domains without constructing
 * processors.
 *
 * Filter, matrix, and nonlinear processor domain coverage remains incomplete. Success does not guarantee that
 * processor preparation will succeed or that the resulting network is numerically stable.
 */
[[nodiscard]] std::expected<void, std::vector<ConfigIssue>> ValidateFDNStructure(const FDNConfig& config);

std::unique_ptr<FDN> CreateFDNFromConfig(const FDNConfig& config);

void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p);
void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p);

} // namespace sfFDN