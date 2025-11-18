#pragma once

#include "sffdn/delay_interp.h"
#include "sffdn/fdn.h"
#include "sffdn/filter_feedback_matrix.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace sfFDN
{

enum class DelayFilterType : uint8_t
{
    Proportional = 0,
    OnePole = 1,
    TwoFilter = 2,
};

using matrix_variant_t = std::variant<sfFDN::CascadedFeedbackMatrixInfo, std::vector<float>>;

struct SchroederAllpassConfig
{
    uint32_t order{0}; // Number of allpass filters per channel
    std::vector<uint32_t> delays;
    std::vector<float> gains;
    bool parallel{false};
};

struct TimeVaryingGainsConfig
{
    float lfo_frequency{0.0f};
    float lfo_amplitude{0.0f};
};

struct TimeVaryingDelayConfig
{
    std::vector<float> lfo_frequencies{0.0f};
    std::vector<float> lfo_amplitudes{0.0f};
    std::vector<float> lfo_initial_phases{0.0f};
    sfFDN::DelayInterpolationType interp_type{sfFDN::DelayInterpolationType::Allpass};
};

struct VelvetNoiseDecorrelatorConfig
{
    std::vector<std::vector<float>> sequence;
};

struct FDNConfig
{
    uint32_t N; // Number of channels
    bool transposed;
    std::vector<float> input_gains;      // Input gains for each channel
    std::vector<float> output_gains;     // Output gains for each channel
    std::vector<uint32_t> delays;        // Delay lengths in samples for each channel
    matrix_variant_t matrix_info;        // Info for feedback matrix
    std::vector<float> attenuation_t60s; // T60 values for attenuation filters
    std::vector<float> tc_gains;         // Tone correction gains for each band
    std::vector<float> tc_frequencies;   // Center frequencies for tone correction bands

    // Extras!
    // Input Stage
    bool use_extra_delays;
    std::vector<uint32_t> input_stage_delays;
    std::optional<VelvetNoiseDecorrelatorConfig> input_velvet_decorrelator = std::nullopt;
    std::optional<VelvetNoiseDecorrelatorConfig> input_velvet_decorrelator_mc = std::nullopt;
    std::optional<SchroederAllpassConfig> input_series_schroeder_config = std::nullopt;
    std::optional<SchroederAllpassConfig> input_schroeder_allpass_config = std::nullopt;
    std::optional<sfFDN::CascadedFeedbackMatrixInfo> input_diffuser = std::nullopt;
    std::optional<TimeVaryingGainsConfig> time_varying_input_gains = std::nullopt;

    // Output Stage
    std::optional<TimeVaryingGainsConfig> time_varying_output_gains = std::nullopt;
    std::optional<SchroederAllpassConfig> output_schroeder_allpass_config = std::nullopt;
    std::optional<VelvetNoiseDecorrelatorConfig> output_velvet_decorrelator = std::nullopt;
    std::optional<VelvetNoiseDecorrelatorConfig> output_velvet_decorrelator_mc = std::nullopt;

    // Feedback Stage
    std::optional<SchroederAllpassConfig> feedback_schroeder_allpass_config = std::nullopt;
    std::optional<TimeVaryingDelayConfig> time_varying_delays = std::nullopt;

    static void LoadFromFile(const std::string& filename, FDNConfig& config);
    static void SaveToFile(const std::string& filename, const FDNConfig& config);
};

std::unique_ptr<sfFDN::FDN> CreateFDNFromConfig(const FDNConfig& config, uint32_t samplerate);
} // namespace sfFDN

void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p);
void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p);
