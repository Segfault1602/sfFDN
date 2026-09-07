#pragma once

#include "sffdn/types.h"
#include "sffdn/fdn_config.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sfFDN
{

namespace json_detail
{
inline void RequireObject(const nlohmann::json& j, const char* name)
{
    if (!j.is_object())
    {
        throw std::invalid_argument(std::string(name) + " must be an object");
    }
}

inline uint32_t ReadUint32(const nlohmann::json& j)
{
    if (j.is_number_unsigned())
    {
        const auto value = j.get<uint64_t>();
        if (value <= std::numeric_limits<uint32_t>::max())
        {
            return static_cast<uint32_t>(value);
        }
    }
    else if (j.is_number_integer())
    {
        const auto value = j.get<int64_t>();
        if (value >= 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max())
        {
            return static_cast<uint32_t>(value);
        }
    }
    throw std::invalid_argument("Expected a uint32 JSON integer");
}

inline float ReadFloat(const nlohmann::json& j)
{
    if (!j.is_number())
    {
        throw std::invalid_argument("Expected a JSON number");
    }
    const double value = j.get<double>();
    if (!std::isfinite(value) || value > std::numeric_limits<float>::max() ||
        value < -std::numeric_limits<float>::max())
    {
        throw std::invalid_argument("JSON number cannot be represented as float");
    }
    return static_cast<float>(value);
}

template <typename T>
T ReadValue(const nlohmann::json& j)
{
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return ReadUint32(j);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return ReadFloat(j);
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        if (!j.is_boolean())
        {
            throw std::invalid_argument("Expected a JSON boolean");
        }
        return j.get<bool>();
    }
    else
    {
        return j.get<T>();
    }
}

template <typename T>
std::vector<T> ReadVector(const nlohmann::json& j)
{
    if (!j.is_array())
    {
        throw std::invalid_argument("Expected a JSON array");
    }
    std::vector<T> result;
    result.reserve(j.size());
    for (const auto& value : j)
    {
        result.push_back(ReadValue<T>(value));
    }
    return result;
}

template <typename T, size_t N>
std::array<T, N> ReadArray(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != N)
    {
        throw std::invalid_argument("Expected a JSON array with the required length");
    }
    std::array<T, N> result{};
    for (size_t index = 0; index < N; ++index)
    {
        result[index] = ReadValue<T>(j[index]);
    }
    return result;
}

inline std::pair<uint32_t, float> ReadSparseFirCoefficient(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != 2)
    {
        throw std::invalid_argument("Sparse FIR coefficient must be a pair");
    }
    return {ReadUint32(j[0]), ReadFloat(j[1])};
}

template <typename T>
void ReadField(const nlohmann::json& j, const char* name, T& value)
{
    value = ReadValue<T>(j.at(name));
}
} // namespace json_detail

#define SFFDN_JSON_ENUM(TYPE, ...)                                                                                     \
    inline void to_json(nlohmann::json& j, const TYPE value)                                                           \
    {                                                                                                                  \
        for (const auto& [enum_value, text] : std::initializer_list<std::pair<TYPE, const char*>>{__VA_ARGS__})        \
        {                                                                                                              \
            if (value == enum_value)                                                                                   \
            {                                                                                                          \
                j = text;                                                                                              \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        throw std::invalid_argument("Unknown " #TYPE " value");                                                        \
    }                                                                                                                  \
    inline void from_json(const nlohmann::json& j, TYPE& value)                                                        \
    {                                                                                                                  \
        if (!j.is_string())                                                                                            \
        {                                                                                                              \
            throw std::invalid_argument("Expected a string for " #TYPE);                                               \
        }                                                                                                              \
        const auto text = j.get<std::string>();                                                                        \
        for (const auto& [enum_value, enum_text] : std::initializer_list<std::pair<TYPE, const char*>>{__VA_ARGS__})   \
        {                                                                                                              \
            if (text == enum_text)                                                                                     \
            {                                                                                                          \
                value = enum_value;                                                                                    \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        throw std::invalid_argument("Unknown " #TYPE " string");                                                       \
    }

SFFDN_JSON_ENUM(ScalarMatrixType, {ScalarMatrixType::Identity, "Identity"}, {ScalarMatrixType::Random, "Random"},
                {ScalarMatrixType::Householder, "Householder"},
                {ScalarMatrixType::RandomHouseholder, "RandomHouseholder"}, {ScalarMatrixType::Hadamard, "Hadamard"},
                {ScalarMatrixType::Circulant, "Circulant"}, {ScalarMatrixType::Allpass, "Allpass"},
                {ScalarMatrixType::NestedAllpass, "NestedAllpass"},
                {ScalarMatrixType::VariableDiffusion, "VariableDiffusion"}, {ScalarMatrixType::Count, "Count"});
SFFDN_JSON_ENUM(DelayInterpolationType, {DelayInterpolationType::None, "None"},
                {DelayInterpolationType::Linear, "Linear"}, {DelayInterpolationType::Allpass, "Allpass"},
                {DelayInterpolationType::Lagrange, "Lagrange"});
SFFDN_JSON_ENUM(DelayLengthType, {DelayLengthType::Random, "Random"}, {DelayLengthType::Gaussian, "Gaussian"},
                {DelayLengthType::Primes, "Primes"}, {DelayLengthType::Uniform, "Uniform"},
                {DelayLengthType::PrimePower, "PrimePower"}, {DelayLengthType::SteamAudio, "SteamAudio"});
SFFDN_JSON_ENUM(ParallelGainsMode, {ParallelGainsMode::Split, "Split"}, {ParallelGainsMode::Merge, "Merge"},
                {ParallelGainsMode::Parallel, "Parallel"});
SFFDN_JSON_ENUM(TimeVaryingMatrixMode, {TimeVaryingMatrixMode::Hadamard, "Hadamard"},
                {TimeVaryingMatrixMode::RealSchur, "RealSchur"}, {TimeVaryingMatrixMode::Count, "Count"});

#undef SFFDN_JSON_ENUM

void to_json(nlohmann::json& j, const VariableDiffusionOptions& config);
void from_json(const nlohmann::json& j, VariableDiffusionOptions& config);
void to_json(nlohmann::json& j, const MatrixGeneratorOptions& config);
void from_json(const nlohmann::json& j, MatrixGeneratorOptions& config);
void to_json(nlohmann::json& j, const GeneratedMatrixOptions& config);
void from_json(const nlohmann::json& j, GeneratedMatrixOptions& config);
void to_json(nlohmann::json& j, const MatrixData& config);
void from_json(const nlohmann::json& j, MatrixData& config);
void to_json(nlohmann::json& j, const ScalarFeedbackMatrixOptions& config);
void from_json(const nlohmann::json& j, ScalarFeedbackMatrixOptions& config);
void to_json(nlohmann::json& j, const CascadedFeedbackMatrixOptions& config);
void from_json(const nlohmann::json& j, CascadedFeedbackMatrixOptions& config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ModulationOptions, frequency, amplitude, initial_phase);
inline void from_json(const nlohmann::json& j, ModulationOptions& config)
{
    json_detail::RequireObject(j, "ModulationOptions");
    ModulationOptions candidate;
    json_detail::ReadField(j, "frequency", candidate.frequency);
    json_detail::ReadField(j, "amplitude", candidate.amplitude);
    json_detail::ReadField(j, "initial_phase", candidate.initial_phase);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(TimeVaryingFeedbackMatrixOptions, matrix_size, mode,
                                                  time_varying_config, rng_seed);
inline void from_json(const nlohmann::json& j, TimeVaryingFeedbackMatrixOptions& config)
{
    json_detail::RequireObject(j, "TimeVaryingFeedbackMatrixOptions");
    TimeVaryingFeedbackMatrixOptions candidate;
    json_detail::ReadField(j, "matrix_size", candidate.matrix_size);
    json_detail::ReadField(j, "mode", candidate.mode);
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    json_detail::ReadField(j, "rng_seed", candidate.rng_seed);
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ParallelGainsOptions, mode, gains, time_varying_config);
inline void from_json(const nlohmann::json& j, ParallelGainsOptions& config)
{
    json_detail::RequireObject(j, "ParallelGainsOptions");
    ParallelGainsOptions candidate;
    json_detail::ReadField(j, "mode", candidate.mode);
    candidate.gains = json_detail::ReadVector<float>(j.at("gains"));
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    config = std::move(candidate);
}
inline void to_json(nlohmann::json& j, const StageGainsOptions& config)
{
    j = {{"gains", config.gains}, {"time_varying_config", config.time_varying_config}};
}
inline void from_json(const nlohmann::json& j, StageGainsOptions& config)
{
    json_detail::RequireObject(j, "StageGainsOptions");
    if (j.size() != 2 || !j.contains("gains") || !j.contains("time_varying_config"))
    {
        throw std::invalid_argument("StageGainsOptions must contain exactly gains and time_varying_config");
    }
    StageGainsOptions candidate;
    candidate.gains = json_detail::ReadVector<float>(j.at("gains"));
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    config = std::move(candidate);
}
void to_json(nlohmann::json& j, const DelayOptions& config);
void from_json(const nlohmann::json& j, DelayOptions& config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(DelayBankOptions, delays, block_size, interpolation_type);
inline void from_json(const nlohmann::json& j, DelayBankOptions& config)
{
    json_detail::RequireObject(j, "DelayBankOptions");
    DelayBankOptions candidate;
    candidate.delays = json_detail::ReadVector<float>(j.at("delays"));
    json_detail::ReadField(j, "block_size", candidate.block_size);
    json_detail::ReadField(j, "interpolation_type", candidate.interpolation_type);
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(DelayBankTimeVaryingOptions, delays, max_delay, interpolation_type,
                                                  time_varying_config);
inline void from_json(const nlohmann::json& j, DelayBankTimeVaryingOptions& config)
{
    json_detail::RequireObject(j, "DelayBankTimeVaryingOptions");
    DelayBankTimeVaryingOptions candidate;
    candidate.delays = json_detail::ReadVector<float>(j.at("delays"));
    json_detail::ReadField(j, "max_delay", candidate.max_delay);
    json_detail::ReadField(j, "interpolation_type", candidate.interpolation_type);
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(FilterCoefficients, b0, b1, b2, a0, a1, a2);
inline void from_json(const nlohmann::json& j, FilterCoefficients& config)
{
    json_detail::RequireObject(j, "FilterCoefficients");
    FilterCoefficients candidate{};
    json_detail::ReadField(j, "b0", candidate.b0);
    json_detail::ReadField(j, "b1", candidate.b1);
    json_detail::ReadField(j, "b2", candidate.b2);
    json_detail::ReadField(j, "a0", candidate.a0);
    json_detail::ReadField(j, "a1", candidate.a1);
    json_detail::ReadField(j, "a2", candidate.a2);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(AllpassFilterOptions, coeff);
inline void from_json(const nlohmann::json& j, AllpassFilterOptions& config)
{
    json_detail::RequireObject(j, "AllpassFilterOptions");
    AllpassFilterOptions candidate;
    json_detail::ReadField(j, "coeff", candidate.coeff);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(SparseFirOptions, coeffs);
inline void from_json(const nlohmann::json& j, SparseFirOptions& config)
{
    json_detail::RequireObject(j, "SparseFirOptions");
    SparseFirOptions candidate;
    const auto& coeffs = j.at("coeffs");
    if (!coeffs.is_array())
    {
        throw std::invalid_argument("SparseFirOptions coeffs must be an array");
    }
    candidate.coeffs.reserve(coeffs.size());
    for (const auto& coefficient : coeffs)
    {
        candidate.coeffs.push_back(json_detail::ReadSparseFirCoefficient(coefficient));
    }
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(CascadedBiquadsOptions, coeffs);
inline void from_json(const nlohmann::json& j, CascadedBiquadsOptions& config)
{
    json_detail::RequireObject(j, "CascadedBiquadsOptions");
    CascadedBiquadsOptions candidate;
    candidate.coeffs = json_detail::ReadVector<FilterCoefficients>(j.at("coeffs"));
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(FirOptions, coeffs);
inline void from_json(const nlohmann::json& j, FirOptions& config)
{
    json_detail::RequireObject(j, "FirOptions");
    FirOptions candidate;
    candidate.coeffs = json_detail::ReadVector<float>(j.at("coeffs"));
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(SchroederAllpassSectionOptions, delays, gains, parallel);
inline void from_json(const nlohmann::json& j, SchroederAllpassSectionOptions& config)
{
    json_detail::RequireObject(j, "SchroederAllpassSectionOptions");
    SchroederAllpassSectionOptions candidate;
    candidate.delays = json_detail::ReadVector<float>(j.at("delays"));
    candidate.gains = json_detail::ReadVector<float>(j.at("gains"));
    json_detail::ReadField(j, "parallel", candidate.parallel);
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(TimeVaryingSchroederAllpassSectionOptions, delays, gains,
                                                  time_varying_config, parallel);
inline void from_json(const nlohmann::json& j, TimeVaryingSchroederAllpassSectionOptions& config)
{
    json_detail::RequireObject(j, "TimeVaryingSchroederAllpassSectionOptions");
    TimeVaryingSchroederAllpassSectionOptions candidate;
    candidate.delays = json_detail::ReadVector<float>(j.at("delays"));
    candidate.gains = json_detail::ReadVector<float>(j.at("gains"));
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    json_detail::ReadField(j, "parallel", candidate.parallel);
    config = std::move(candidate);
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(DattorroDelayOptions, delay_config, blend, feedforward, feedback);
inline void from_json(const nlohmann::json& j, DattorroDelayOptions& config)
{
    json_detail::RequireObject(j, "DattorroDelayOptions");
    DattorroDelayOptions candidate;
    json_detail::ReadField(j, "delay_config", candidate.delay_config);
    json_detail::ReadField(j, "blend", candidate.blend);
    json_detail::ReadField(j, "feedforward", candidate.feedforward);
    json_detail::ReadField(j, "feedback", candidate.feedback);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ControllableFullWaveRectifierOptions, alpha, antialiasing, dc_block,
                                                  sample_rate);
inline void from_json(const nlohmann::json& j, ControllableFullWaveRectifierOptions& config)
{
    json_detail::RequireObject(j, "ControllableFullWaveRectifierOptions");
    ControllableFullWaveRectifierOptions candidate;
    json_detail::ReadField(j, "alpha", candidate.alpha);
    json_detail::ReadField(j, "antialiasing", candidate.antialiasing);
    json_detail::ReadField(j, "dc_block", candidate.dc_block);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(SignalDependentFractionalDelayOptions, d);
inline void from_json(const nlohmann::json& j, SignalDependentFractionalDelayOptions& config)
{
    json_detail::RequireObject(j, "SignalDependentFractionalDelayOptions");
    SignalDependentFractionalDelayOptions candidate;
    json_detail::ReadField(j, "d", candidate.d);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(RingModulatorOptions, frequency, amplitude, initial_phase);
inline void from_json(const nlohmann::json& j, RingModulatorOptions& config)
{
    json_detail::RequireObject(j, "RingModulatorOptions");
    RingModulatorOptions candidate;
    json_detail::ReadField(j, "frequency", candidate.frequency);
    json_detail::ReadField(j, "amplitude", candidate.amplitude);
    json_detail::ReadField(j, "initial_phase", candidate.initial_phase);
    config = candidate;
}
void to_json(nlohmann::json& j, const MultichannelProcessorOptions& config);
void from_json(const nlohmann::json& j, MultichannelProcessorOptions& config);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(HomogenousFilterOptions, t60, delay, sample_rate);
inline void from_json(const nlohmann::json& j, HomogenousFilterOptions& config)
{
    json_detail::RequireObject(j, "HomogenousFilterOptions");
    HomogenousFilterOptions candidate;
    json_detail::ReadField(j, "t60", candidate.t60);
    json_detail::ReadField(j, "delay", candidate.delay);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(TwoBandFilterOptions, t60s, delay, sample_rate);
inline void from_json(const nlohmann::json& j, TwoBandFilterOptions& config)
{
    json_detail::RequireObject(j, "TwoBandFilterOptions");
    TwoBandFilterOptions candidate;
    candidate.t60s = json_detail::ReadArray<float, 2>(j.at("t60s"));
    json_detail::ReadField(j, "delay", candidate.delay);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ThreeBandFilterOptions, t60s, delay, freqs, q, sample_rate);
inline void from_json(const nlohmann::json& j, ThreeBandFilterOptions& config)
{
    json_detail::RequireObject(j, "ThreeBandFilterOptions");
    ThreeBandFilterOptions candidate;
    candidate.t60s = json_detail::ReadArray<float, 3>(j.at("t60s"));
    json_detail::ReadField(j, "delay", candidate.delay);
    candidate.freqs = json_detail::ReadArray<float, 2>(j.at("freqs"));
    json_detail::ReadField(j, "q", candidate.q);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(TenBandFilterOptions, t60s, delay, sample_rate, shelf_cutoff);
inline void from_json(const nlohmann::json& j, TenBandFilterOptions& config)
{
    json_detail::RequireObject(j, "TenBandFilterOptions");
    TenBandFilterOptions candidate;
    candidate.t60s = json_detail::ReadArray<float, 10>(j.at("t60s"));
    json_detail::ReadField(j, "delay", candidate.delay);
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    json_detail::ReadField(j, "shelf_cutoff", candidate.shelf_cutoff);
    config = candidate;
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(GraphicEQOptions, gains_db, freqs, sample_rate);
inline void from_json(const nlohmann::json& j, GraphicEQOptions& config)
{
    json_detail::RequireObject(j, "GraphicEQOptions");
    GraphicEQOptions candidate;
    candidate.gains_db = json_detail::ReadArray<float, 10>(j.at("gains_db"));
    candidate.freqs = json_detail::ReadArray<float, 10>(j.at("freqs"));
    json_detail::ReadField(j, "sample_rate", candidate.sample_rate);
    config = candidate;
}

void to_json(nlohmann::json& j, const AttenuationFilterBankOptions& config);
void from_json(const nlohmann::json& j, AttenuationFilterBankOptions& config);

void to_json(nlohmann::json& j, const InputStageConfig& p);
void from_json(const nlohmann::json& j, InputStageConfig& p);
void to_json(nlohmann::json& j, const OutputStageConfig& p);
void from_json(const nlohmann::json& j, OutputStageConfig& p);
void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p);
void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p);

} // namespace sfFDN
