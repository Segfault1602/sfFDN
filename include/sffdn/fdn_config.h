#pragma once

#include "sffdn/config_diagnostics.h"
#include "sffdn/types.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace sfFDN
{

class FDN;

/** @brief Configuration for FDN input-stage processing. */
struct InputStageConfig
{
    //! Single-channel processors applied before the signal is split into FDN channels.
    std::vector<single_channel_processor_variant_t> single_channel_processors;

    //! Stage gains. Routing is determined by this stage's placement.
    StageGainsOptions parallel_gains_config;

    //! Multi-channel processors applied after the stage gains.
    std::vector<multi_channel_processor_variant_t> multichannel_processors;

    bool operator==(const InputStageConfig&) const = default;
};

/** @brief Configuration for FDN output-stage processing. */
struct OutputStageConfig
{
    //! Multi-channel processors applied before the FDN channels are mixed down.
    std::vector<multi_channel_processor_variant_t> multichannel_processors;

    //! Stage gains. Routing is determined by this stage's placement.
    StageGainsOptions parallel_gains_config;

    //! Single-channel processors applied after the FDN channels are mixed down.
    std::vector<single_channel_processor_variant_t> single_channel_processors;

    bool operator==(const OutputStageConfig&) const = default;
};

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

    //! Sample rate associated with this configuration; filter options specify their own rates.
    float sample_rate{static_cast<float>(kDefaultSampleRate)};

    //! Delay bank configuration. Its block size must be nonzero and at least this configuration's block size.
    DelayBankOptions delay_bank_config;

    //! Input gain block.
    InputStageConfig input_block_config;

    //! Feedback matrix block
    feedback_matrix_variant_t feedback_matrix_config;

    //! Attenuation filter bank block
    std::optional<AttenuationFilterBankOptions> attenuation_filter_bank_config;

    //! Loop filter block. A MultichannelProcessorOptions bank has exactly fdn_size channels.
    std::vector<multi_channel_processor_variant_t> loop_filter_configs;

    //! Output gain block.
    OutputStageConfig output_block_config;

    //! Tone correction filter block
    std::vector<single_channel_processor_variant_t> tone_correction_filters;

    bool operator==(const FDNConfig&) const = default;
};

/** @brief Creates a complete deterministic wet FDN network with one-second homogeneous decay.
 *
 * Uses a Hadamard feedback matrix for power-of-two sizes and Householder otherwise, with normalized gains.
 * @throws std::invalid_argument if `fdn_size` (order) or `block_size` is zero, or `sample_rate` is nonpositive.
 * @pre Inputs are finite and use practical FDN sizes.
 */
[[nodiscard]] FDNConfig MakeDefaultFDNConfig(uint32_t fdn_size = 8U, uint32_t block_size = kDefaultBlockSize,
                                             float sample_rate = static_cast<float>(kDefaultSampleRate));

/** @brief Validates the FDN graph, its dimensions, and every supported processor option domain without constructing
 * processors.
 *
 * This does not perform numerical decomposition or filter design, guarantee allocation success, or certify acoustic
 * stability.
 */
[[nodiscard]] std::expected<void, std::vector<ConfigIssue>> ValidateFDNConfig(const FDNConfig& config);

/** @brief Assigns fresh seeds to every generated matrix recipe in an FDN configuration.
 *
 * Explicit matrix data and all non-matrix configuration values are unchanged.
 */
void RandomizeMatrixSeeds(FDNConfig& config);

std::unique_ptr<FDN> CreateFDNFromConfig(const FDNConfig& config);

} // namespace sfFDN