#include "json_helper.h"

#include "sffdn/types.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sfFDN
{
namespace
{
const nlohmann::json& TaggedValue(const nlohmann::json& j, std::initializer_list<const char*> tags)
{
    if (!j.is_object() || j.size() != 1)
    {
        throw std::invalid_argument("Processor config must contain exactly one processor type");
    }

    const auto entry = j.begin();
    for (const char* tag : tags)
    {
        if (entry.key() == tag)
        {
            return entry.value();
        }
    }
    throw std::invalid_argument("Unknown processor config type");
}

} // namespace

void to_json(nlohmann::json& j, const VariableDiffusionOptions& config)
{
    j = {{"diffusion", config.diffusion}};
}

void from_json(const nlohmann::json& j, VariableDiffusionOptions& config)
{
    if (!j.is_object() || j.size() != 1 || !j.contains("diffusion"))
    {
        throw std::invalid_argument("VariableDiffusionOptions must contain exactly diffusion");
    }
    VariableDiffusionOptions candidate;
    json_detail::ReadField(j, "diffusion", candidate.diffusion);
    config = candidate;
}

void to_json(nlohmann::json& j, const MatrixGeneratorOptions& config)
{
    std::visit(overloaded{
                   [&j](ScalarMatrixType type) { j = type; },
                   [&j](const VariableDiffusionOptions& options) { j = {{"VariableDiffusionOptions", options}}; },
               },
               config);
}

void from_json(const nlohmann::json& j, MatrixGeneratorOptions& config)
{
    MatrixGeneratorOptions candidate;
    if (j.is_string())
    {
        candidate = j.get<ScalarMatrixType>();
    }
    else
    {
        const auto& options = TaggedValue(j, {"VariableDiffusionOptions"});
        candidate = options.get<VariableDiffusionOptions>();
    }
    config = candidate;
}

void to_json(nlohmann::json& j, const GeneratedMatrixOptions& config)
{
    j = {{"matrix_size", config.matrix_size}, {"generator", config.generator}, {"rng_seed", config.rng_seed}};
}

void from_json(const nlohmann::json& j, GeneratedMatrixOptions& config)
{
    if (!j.is_object() || j.size() != 3 || !j.contains("matrix_size") || !j.contains("generator") ||
        !j.contains("rng_seed"))
    {
        throw std::invalid_argument("GeneratedMatrixOptions must contain exactly matrix_size, generator and rng_seed");
    }
    GeneratedMatrixOptions candidate;
    json_detail::ReadField(j, "matrix_size", candidate.matrix_size);
    candidate.generator = j.at("generator").get<MatrixGeneratorOptions>();
    json_detail::ReadField(j, "rng_seed", candidate.rng_seed);
    config = candidate;
}

void to_json(nlohmann::json& j, const MatrixData& config)
{
    const auto values = config.Values();
    j = {{"order", config.Order()}, {"coefficients", std::vector<float>(values.begin(), values.end())}};
}

void from_json(const nlohmann::json& j, MatrixData& config)
{
    if (!j.is_object() || j.size() != 2 || !j.contains("order") || !j.contains("coefficients"))
    {
        throw std::invalid_argument("MatrixData must contain exactly order and coefficients");
    }
    const auto order = json_detail::ReadUint32(j.at("order"));
    auto coefficients = json_detail::ReadVector<float>(j.at("coefficients"));
    MatrixData candidate(order, std::move(coefficients));
    config = std::move(candidate);
}

void to_json(nlohmann::json& j, const ScalarFeedbackMatrixOptions& config)
{
    nlohmann::json source;
    std::visit(overloaded{
                   [&source](const GeneratedMatrixOptions& options) { source = {{"GeneratedMatrixOptions", options}}; },
                   [&source](const MatrixData& data) { source = {{"MatrixData", data}}; },
               },
               config.source);
    j = {{"source", std::move(source)}};
}

void from_json(const nlohmann::json& j, ScalarFeedbackMatrixOptions& config)
{
    if (!j.is_object() || j.size() != 1 || !j.contains("source"))
    {
        throw std::invalid_argument("ScalarFeedbackMatrixOptions must contain exactly source");
    }
    const auto& source = TaggedValue(j.at("source"), {"GeneratedMatrixOptions", "MatrixData"});

    ScalarFeedbackMatrixOptions candidate;
    if (j.at("source").contains("GeneratedMatrixOptions"))
    {
        candidate.source = source.get<GeneratedMatrixOptions>();
    }
    else
    {
        candidate.source = source.get<MatrixData>();
    }
    config = std::move(candidate);
}

void to_json(nlohmann::json& j, const CascadedFeedbackMatrixOptions& config)
{
    j = {
        {"matrix_size", config.matrix_size},
        {"stage_count", config.stage_count},
        {"sparsity", config.sparsity},
        {"generator", config.generator},
        {"gain_per_samples", config.gain_per_samples},
        {"rng_seed", config.rng_seed},
    };
}

void from_json(const nlohmann::json& j, CascadedFeedbackMatrixOptions& config)
{
    if (!j.is_object() || j.size() != 6 || !j.contains("matrix_size") || !j.contains("stage_count") ||
        !j.contains("sparsity") || !j.contains("generator") || !j.contains("gain_per_samples") ||
        !j.contains("rng_seed"))
    {
        throw std::invalid_argument(
            "CascadedFeedbackMatrixOptions must contain exactly matrix_size, stage_count, sparsity, generator, "
            "gain_per_samples and rng_seed");
    }
    CascadedFeedbackMatrixOptions candidate;
    json_detail::ReadField(j, "matrix_size", candidate.matrix_size);
    json_detail::ReadField(j, "stage_count", candidate.stage_count);
    json_detail::ReadField(j, "sparsity", candidate.sparsity);
    candidate.generator = j.at("generator").get<MatrixGeneratorOptions>();
    json_detail::ReadField(j, "gain_per_samples", candidate.gain_per_samples);
    json_detail::ReadField(j, "rng_seed", candidate.rng_seed);
    config = candidate;
}

void to_json(nlohmann::json& j, const DelayOptions& config)
{
    j["delay"] = config.delay;
    j["max_delay"] = config.max_delay;
    j["interp_type"] = config.interp_type;
    if (config.lfo_config.has_value())
    {
        j["lfo_config"] = config.lfo_config.value();
    }
}

void from_json(const nlohmann::json& j, DelayOptions& config)
{
    json_detail::RequireObject(j, "DelayOptions");
    DelayOptions candidate;
    json_detail::ReadField(j, "delay", candidate.delay);
    json_detail::ReadField(j, "max_delay", candidate.max_delay);
    json_detail::ReadField(j, "interp_type", candidate.interp_type);
    if (j.contains("lfo_config") && !j["lfo_config"].is_null())
    {
        candidate.lfo_config = j["lfo_config"].get<ModulationOptions>();
    }
    config = candidate;
}

void to_json(nlohmann::json& j, const AttenuationFilterBankOptions& config)
{
    nlohmann::json filter_configs_json = nlohmann::json::array();
    for (const auto& filter_config : config.filter_configs)
    {
        filter_configs_json.push_back(std::visit(overloaded{
                                                     [](const HomogenousFilterOptions& config) {
                                                         nlohmann::json j;
                                                         j["ProportionalAttenuationConfig"] = config;
                                                         return j;
                                                     },
                                                     [](const TwoBandFilterOptions& config) {
                                                         nlohmann::json j;
                                                         j["TwoBandFilterConfig"] = config;
                                                         return j;
                                                     },
                                                     [](const ThreeBandFilterOptions& config) {
                                                         nlohmann::json j;
                                                         j["ThreeBandFilterConfig"] = config;
                                                         return j;
                                                     },
                                                     [](const TenBandFilterOptions& config) {
                                                         nlohmann::json j;
                                                         j["TenBandFilterConfig"] = config;
                                                         return j;
                                                     },
                                                 },
                                                 filter_config));
    }
    j["AttenuationFilterBankOptions"] = filter_configs_json;
}

void from_json(const nlohmann::json& j, AttenuationFilterBankOptions& config)
{
    if (!j.is_array())
    {
        throw std::invalid_argument("AttenuationFilterBankOptions must be an array.");
    }

    AttenuationFilterBankOptions candidate;
    for (const auto& filter_config_json : j)
    {
        const auto& filter_config = TaggedValue(filter_config_json, {
                                                                        "ProportionalAttenuationConfig",
                                                                        "TwoBandFilterConfig",
                                                                        "ThreeBandFilterConfig",
                                                                        "TenBandFilterConfig",
                                                                    });
        if (filter_config_json.contains("ProportionalAttenuationConfig"))
        {
            candidate.filter_configs.emplace_back(filter_config.get<HomogenousFilterOptions>());
        }
        else if (filter_config_json.contains("TwoBandFilterConfig"))
        {
            candidate.filter_configs.emplace_back(filter_config.get<TwoBandFilterOptions>());
        }
        else if (filter_config_json.contains("ThreeBandFilterConfig"))
        {
            candidate.filter_configs.emplace_back(filter_config.get<ThreeBandFilterOptions>());
        }
        else if (filter_config_json.contains("TenBandFilterConfig"))
        {
            candidate.filter_configs.emplace_back(filter_config.get<TenBandFilterOptions>());
        }
        else
        {
            throw std::invalid_argument("Unknown filter config type in AttenuationFilterBankOptions");
        }
    }
    config = std::move(candidate);
}

nlohmann::json ToJson(const feedback_matrix_variant_t& matrix_config)
{
    return std::visit(overloaded{
                          [](const CascadedFeedbackMatrixOptions& info) {
                              nlohmann::json mat;
                              mat["CascadedFeedbackMatrixInfo"] = info;
                              return mat;
                          },
                          [](const ScalarFeedbackMatrixOptions& config) {
                              nlohmann::json mat;
                              mat["ScalarFeedbackMatrixOptions"] = config;
                              return mat;
                          },
                          [](const TimeVaryingFeedbackMatrixOptions& config) {
                              nlohmann::json mat;
                              mat["TimeVaryingFeedbackMatrixOptions"] = config;
                              return mat;
                          },
                          [](const KroneckerFeedbackMatrixOptions& config) {
                              nlohmann::json mat;
                              mat["KroneckerFeedbackMatrixOptions"] = config;
                              return mat;
                          },
                          [](const TimeVaryingKroneckerFeedbackMatrixOptions& config) {
                              nlohmann::json mat;
                              mat["TimeVaryingKroneckerFeedbackMatrixOptions"] = config;
                              return mat;
                          },
                      },
                      matrix_config);
}

void to_json(nlohmann::json& j, const KroneckerFeedbackMatrixOptions& config)
{
    j = {
        {"matrix_size", config.matrix_size},
        {"angles", config.angles},
        {"kernel_types", config.kernel_types},
    };
}

void from_json(const nlohmann::json& j, KroneckerFeedbackMatrixOptions& config)
{
    if (!j.is_object() || j.size() != 3 || !j.contains("matrix_size") || !j.contains("angles") ||
        !j.contains("kernel_types"))
    {
        throw std::invalid_argument("KroneckerFeedbackMatrixOptions must contain exactly matrix_size, angles and "
                                    "kernel_types");
    }
    KroneckerFeedbackMatrixOptions candidate;
    json_detail::ReadField(j, "matrix_size", candidate.matrix_size);
    candidate.angles = json_detail::ReadVector<float>(j.at("angles"));
    candidate.kernel_types = json_detail::ReadVector<KroneckerKernelType>(j.at("kernel_types"));
    config = std::move(candidate);
}

void to_json(nlohmann::json& j, const TimeVaryingKroneckerFeedbackMatrixOptions& config)
{
    j = {
        {"matrix", config.matrix},
        {"time_varying_config", config.time_varying_config},
    };
}

void from_json(const nlohmann::json& j, TimeVaryingKroneckerFeedbackMatrixOptions& config)
{
    if (!j.is_object() || j.size() != 2 || !j.contains("matrix") || !j.contains("time_varying_config"))
    {
        throw std::invalid_argument(
            "TimeVaryingKroneckerFeedbackMatrixOptions must contain exactly matrix and time_varying_config");
    }
    TimeVaryingKroneckerFeedbackMatrixOptions candidate;
    candidate.matrix = j.at("matrix").get<KroneckerFeedbackMatrixOptions>();
    candidate.time_varying_config = json_detail::ReadVector<ModulationOptions>(j.at("time_varying_config"));
    config = std::move(candidate);
}

nlohmann::json ToJson(const single_channel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{
                          [](const SchroederAllpassSectionOptions& config) {
                              nlohmann::json proc;
                              proc["SchroederAllpassSectionOptions"] = config;
                              return proc;
                          },
                          [](const TimeVaryingSchroederAllpassSectionOptions& config) {
                              nlohmann::json proc;
                              proc["TimeVaryingSchroederAllpassSectionOptions"] = config;
                              return proc;
                          },
                          [](const AllpassFilterOptions& config) {
                              nlohmann::json proc;
                              proc["AllpassFilterOptions"] = config;
                              return proc;
                          },
                          [](const CascadedBiquadsOptions& config) {
                              nlohmann::json proc;
                              proc["CascadedBiquadsOptions"] = config;
                              return proc;
                          },
                          [](const FirOptions& config) {
                              nlohmann::json proc;
                              proc["FirOptions"] = config;
                              return proc;
                          },
                          [](const DelayOptions& config) {
                              nlohmann::json proc;
                              proc["DelayOptions"] = config;
                              return proc;
                          },
                          [](const DattorroDelayOptions& config) {
                              nlohmann::json proc;
                              proc["DattorroDelayOptions"] = config;
                              return proc;
                          },
                          [](const ControllableFullWaveRectifierOptions& config) {
                              nlohmann::json proc;
                              proc["ControllableFullWaveRectifierOptions"] = config;
                              return proc;
                          },
                          [](const SignalDependentFractionalDelayOptions& config) {
                              nlohmann::json proc;
                              proc["SignalDependentFractionalDelayOptions"] = config;
                              return proc;
                          },
                          [](const RingModulatorOptions& config) {
                              nlohmann::json proc;
                              proc["RingModulatorOptions"] = config;
                              return proc;
                          },
                          [](const GraphicEQOptions& config) {
                              nlohmann::json proc;
                              proc["GraphicEQOptions"] = config;
                              return proc;
                          },
                      },
                      processor_config);
}

nlohmann::json ToJson(const multi_channel_processor_variant_t& processor_config)
{
    return std::visit(overloaded{
                          [](const ParallelGainsOptions& config) {
                              nlohmann::json proc;
                              proc["ParallelGainsConfig"] = config;
                              return proc;
                          },
                          [](const MultichannelProcessorOptions& config) {
                              nlohmann::json proc;
                              proc["MultichannelProcessorOptions"] = config;
                              return proc;
                          },
                          [](const AttenuationFilterBankOptions& config) {
                              nlohmann::json proc = config;
                              return proc;
                          },
                          [](const DelayBankOptions& config) {
                              nlohmann::json proc;
                              proc["DelayBankOptions"] = config;
                              return proc;
                          },
                          [](const DelayBankTimeVaryingOptions& config) {
                              nlohmann::json proc;
                              proc["DelayBankTimeVaryingOptions"] = config;
                              return proc;
                          },
                          [](const CascadedFeedbackMatrixOptions& config) {
                              nlohmann::json proc;
                              proc["CascadedFeedbackMatrixInfo"] = config;
                              return proc;
                          },
                          [](const ScalarFeedbackMatrixOptions& config) {
                              nlohmann::json proc;
                              proc["ScalarFeedbackMatrixOptions"] = config;
                              return proc;
                          },
                          [](const KroneckerFeedbackMatrixOptions& config) {
                              nlohmann::json proc;
                              proc["KroneckerFeedbackMatrixOptions"] = config;
                              return proc;
                          },
                          [](const TimeVaryingKroneckerFeedbackMatrixOptions& config) {
                              nlohmann::json proc;
                              proc["TimeVaryingKroneckerFeedbackMatrixOptions"] = config;
                              return proc;
                          },
                      },
                      processor_config);
}

single_channel_processor_variant_t SingleChannelProcessorFromJson(const nlohmann::json& j)
{
    TaggedValue(j, {
                       "SchroederAllpassSectionOptions",
                       "TimeVaryingSchroederAllpassSectionOptions",
                       "AllpassFilterOptions",
                       "CascadedBiquadsOptions",
                       "FirOptions",
                       "DelayOptions",
                       "GraphicEQOptions",
                       "DattorroDelayOptions",
                       "ControllableFullWaveRectifierOptions",
                       "SignalDependentFractionalDelayOptions",
                       "RingModulatorOptions",
                   });
    if (j.contains("SchroederAllpassSectionOptions"))
    {
        return j["SchroederAllpassSectionOptions"].get<SchroederAllpassSectionOptions>();
    }

    if (j.contains("TimeVaryingSchroederAllpassSectionOptions"))
    {
        return j["TimeVaryingSchroederAllpassSectionOptions"].get<TimeVaryingSchroederAllpassSectionOptions>();
    }

    if (j.contains("AllpassFilterOptions"))
    {
        return j["AllpassFilterOptions"].get<AllpassFilterOptions>();
    }

    if (j.contains("CascadedBiquadsOptions"))
    {
        return j["CascadedBiquadsOptions"].get<CascadedBiquadsOptions>();
    }

    if (j.contains("FirOptions"))
    {
        return j["FirOptions"].get<FirOptions>();
    }

    if (j.contains("DelayOptions"))
    {
        return j["DelayOptions"].get<DelayOptions>();
    }

    if (j.contains("GraphicEQOptions"))
    {
        return j["GraphicEQOptions"].get<GraphicEQOptions>();
    }

    if (j.contains("DattorroDelayOptions"))
    {
        return j["DattorroDelayOptions"].get<DattorroDelayOptions>();
    }

    if (j.contains("ControllableFullWaveRectifierOptions"))
    {
        return j["ControllableFullWaveRectifierOptions"].get<ControllableFullWaveRectifierOptions>();
    }

    if (j.contains("SignalDependentFractionalDelayOptions"))
    {
        return j["SignalDependentFractionalDelayOptions"].get<SignalDependentFractionalDelayOptions>();
    }

    if (j.contains("RingModulatorOptions"))
    {
        return j["RingModulatorOptions"].get<RingModulatorOptions>();
    }

    throw std::invalid_argument("Unknown single channel processor config type: " + j.dump());
}

multi_channel_processor_variant_t MultichannelProcessorFromJson(const nlohmann::json& j)
{
    TaggedValue(j, {
                       "ParallelGainsConfig",
                       "MultichannelProcessorOptions",
                       "AttenuationFilterBankOptions",
                       "DelayBankOptions",
                       "DelayBankTimeVaryingOptions",
                       "CascadedFeedbackMatrixInfo",
                       "ScalarFeedbackMatrixOptions",
                       "KroneckerFeedbackMatrixOptions",
                       "TimeVaryingKroneckerFeedbackMatrixOptions",
                   });

    if (j.contains("ParallelGainsConfig"))
    {
        auto config = j["ParallelGainsConfig"].get<ParallelGainsOptions>();
        return config;
    }

    if (j.contains("MultichannelProcessorOptions"))
    {
        return j["MultichannelProcessorOptions"].get<MultichannelProcessorOptions>();
    }

    if (j.contains("AttenuationFilterBankOptions"))
    {
        auto config = j["AttenuationFilterBankOptions"].get<AttenuationFilterBankOptions>();
        return config;
    }

    if (j.contains("DelayBankOptions"))
    {
        return j["DelayBankOptions"].get<DelayBankOptions>();
    }

    if (j.contains("DelayBankTimeVaryingOptions"))
    {
        return j["DelayBankTimeVaryingOptions"].get<DelayBankTimeVaryingOptions>();
    }

    if (j.contains("CascadedFeedbackMatrixInfo"))
    {
        return j["CascadedFeedbackMatrixInfo"].get<CascadedFeedbackMatrixOptions>();
    }

    if (j.contains("ScalarFeedbackMatrixOptions"))
    {
        return j["ScalarFeedbackMatrixOptions"].get<ScalarFeedbackMatrixOptions>();
    }

    if (j.contains("KroneckerFeedbackMatrixOptions"))
    {
        return j["KroneckerFeedbackMatrixOptions"].get<KroneckerFeedbackMatrixOptions>();
    }

    if (j.contains("TimeVaryingKroneckerFeedbackMatrixOptions"))
    {
        return j["TimeVaryingKroneckerFeedbackMatrixOptions"].get<TimeVaryingKroneckerFeedbackMatrixOptions>();
    }

    throw std::invalid_argument("Unknown multichannel processor config type");
}

void to_json(nlohmann::json& j, const MultichannelProcessorOptions& config)
{
    j = nlohmann::json{{"channels", nlohmann::json::array()}};
    for (const auto& channel : config.channels)
    {
        j["channels"].push_back(channel.has_value() ? ToJson(channel.value()) : nlohmann::json(nullptr));
    }
}

void from_json(const nlohmann::json& j, MultichannelProcessorOptions& config)
{
    if (!j.is_object() || j.size() != 1 || !j.contains("channels"))
    {
        throw std::invalid_argument("Multichannel processor options must contain exactly a channels array");
    }

    const auto& channels = j.at("channels");
    if (!channels.is_array())
    {
        throw std::invalid_argument("Multichannel processor channels must be an array");
    }

    MultichannelProcessorOptions candidate;
    candidate.channels.reserve(channels.size());
    for (const auto& channel : channels)
    {
        if (channel.is_null())
        {
            candidate.channels.emplace_back(std::nullopt);
        }
        else
        {
            if (!channel.is_object())
            {
                throw std::invalid_argument("Multichannel processor channel must be an object or null");
            }
            if (channel.size() != 1)
            {
                throw std::invalid_argument(
                    "Multichannel processor channel must contain exactly one processor type");
            }
            candidate.channels.emplace_back(SingleChannelProcessorFromJson(channel));
        }
    }
    config = std::move(candidate);
}

feedback_matrix_variant_t FeedbackMatrixFromJson(const nlohmann::json& j)
{
    TaggedValue(j, {"CascadedFeedbackMatrixInfo", "ScalarFeedbackMatrixOptions", "TimeVaryingFeedbackMatrixOptions",
                    "KroneckerFeedbackMatrixOptions", "TimeVaryingKroneckerFeedbackMatrixOptions"});
    if (j.contains("CascadedFeedbackMatrixInfo"))
    {
        auto config = j["CascadedFeedbackMatrixInfo"].get<CascadedFeedbackMatrixOptions>();
        return config;
    }

    if (j.contains("ScalarFeedbackMatrixOptions"))
    {
        ScalarFeedbackMatrixOptions config;
        from_json(j["ScalarFeedbackMatrixOptions"], config);
        return config;
    }

    if (j.contains("TimeVaryingFeedbackMatrixOptions"))
    {
        return j["TimeVaryingFeedbackMatrixOptions"].get<TimeVaryingFeedbackMatrixOptions>();
    }

    if (j.contains("KroneckerFeedbackMatrixOptions"))
    {
        return j["KroneckerFeedbackMatrixOptions"].get<KroneckerFeedbackMatrixOptions>();
    }

    if (j.contains("TimeVaryingKroneckerFeedbackMatrixOptions"))
    {
        return j["TimeVaryingKroneckerFeedbackMatrixOptions"].get<TimeVaryingKroneckerFeedbackMatrixOptions>();
    }

    throw std::invalid_argument("Unknown feedback matrix config type");
}

} // namespace sfFDN