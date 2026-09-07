#include "processor_option_validation.h"

#include "sffdn/config_diagnostics.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/types.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
using sfFDN::ConfigErrorCode;
using sfFDN::ConfigIssue;
using Issues = std::vector<ConfigIssue>;

constexpr uint32_t kDefaultBlockSize = 128U;

void AddIssue(Issues& issues, ConfigErrorCode code, const std::string& path, const char* message)
{
    issues.push_back({.code = code, .path = path, .message = message});
}

std::string IndexPath(const std::string& path, size_t index)
{
    return path + "/" + std::to_string(index);
}

bool IsSupportedScalarType(sfFDN::ScalarMatrixType type)
{
    return type >= sfFDN::ScalarMatrixType::Identity && type < sfFDN::ScalarMatrixType::Count;
}

bool IsSupportedTimeVaryingMode(sfFDN::TimeVaryingMatrixMode mode)
{
    return mode == sfFDN::TimeVaryingMatrixMode::Hadamard || mode == sfFDN::TimeVaryingMatrixMode::RealSchur;
}

bool IsValidScalarDimension(uint32_t size, sfFDN::ScalarMatrixType type)
{
    if (size == 0U)
    {
        return false;
    }
    if (type == sfFDN::ScalarMatrixType::Hadamard)
    {
        return std::has_single_bit(size);
    }
    if (type == sfFDN::ScalarMatrixType::Allpass)
    {
        return size >= 2U && (size % 2U) == 0U;
    }
    if (type == sfFDN::ScalarMatrixType::VariableDiffusion)
    {
        return size >= 2U && std::has_single_bit(size);
    }
    return true;
}

void ValidateModulationOptions(const std::vector<sfFDN::ModulationOptions>& modulations, const std::string& path,
                               Issues& issues)
{
    for (size_t index = 0; index < modulations.size(); ++index)
    {
        const auto& modulation = modulations[index];
        const std::string modulation_path = IndexPath(path, index);
        sfFDN::detail::ValidateModulation(modulation, modulation_path, issues);
        if (std::abs(modulation.amplitude) > 1.0F)
        {
            AddIssue(issues, ConfigErrorCode::InvalidValue, modulation_path + "/amplitude",
                     "amplitude must be in [-1, 1]");
        }
    }
}

bool CascadedShiftFits(uint32_t matrix_size, uint32_t stage_count, float sparsity)
{
    if (stage_count == 0U)
    {
        return true;
    }

    const double log_shift_bound = std::log(static_cast<double>(sparsity)) +
                                   static_cast<double>(stage_count) * std::log(static_cast<double>(matrix_size));
    const double log_delay_capacity =
        std::log(static_cast<double>(std::numeric_limits<uint32_t>::max() - (2U * kDefaultBlockSize)));
    return log_shift_bound <= log_delay_capacity;
}

bool CascadedGainCanOverflow(uint32_t matrix_size, uint32_t stage_count, float sparsity, float gain_per_samples)
{
    const double magnitude = std::abs(static_cast<double>(gain_per_samples));
    if (stage_count == 0U || magnitude <= 1.0)
    {
        return false;
    }

    const double log_shift_bound = std::log(static_cast<double>(sparsity)) +
                                   static_cast<double>(stage_count) * std::log(static_cast<double>(matrix_size));
    const double log_float_max = std::log(static_cast<double>(std::numeric_limits<float>::max()));
    const double log_gain = std::log(magnitude);
    const double maximum_safe_shift = log_float_max / log_gain;
    return log_shift_bound > std::log(maximum_safe_shift);
}

bool CascadedNegativeGainHasFractionalExponents(uint32_t matrix_size, uint32_t stage_count, float sparsity,
                                                float gain_per_samples)
{
    if (gain_per_samples >= 0.0F || stage_count <= 1U || matrix_size <= 1U)
    {
        return false;
    }

    // Later pulse sizes remain integral when this first pulse is integral because they are only multiplied by size.
    const float first_pulse = static_cast<float>(matrix_size) * sparsity;
    return std::trunc(first_pulse) != first_pulse;
}
} // namespace

namespace sfFDN::detail
{
void ValidateOptions(const ScalarFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    std::visit(overloaded{
                   [&](const GeneratedMatrixOptions& source) {
                       ValidateOptions(source, path + "/source/GeneratedMatrixOptions", issues);
                   },
                   [&](const MatrixData& source) {
                       if (source.Order() == 0U)
                       {
                           AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/source/MatrixData/order",
                                    "matrix order must be positive");
                       }
                   }},
               options.source);
}

void ValidateOptions(const GeneratedMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    const std::string matrix_size_path = path + "/matrix_size";
    const bool size_valid = options.matrix_size > 0U;
    if (!size_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, matrix_size_path, "matrix size must be positive");
    }

    const ScalarMatrixType type = GetMatrixType(options.generator);
    if (!IsSupportedScalarType(type))
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, path + "/generator", "matrix type is unsupported");
        return;
    }

    if (size_valid && !IsValidScalarDimension(options.matrix_size, type))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, matrix_size_path,
                 "matrix size is unsupported by the selected matrix type");
    }

    std::visit(overloaded{[](ScalarMatrixType) {},
                          [&](const VariableDiffusionOptions& generator) {
                              ValidateOptions(generator, path + "/generator/VariableDiffusionOptions", issues);
                          }},
               options.generator);
}

void ValidateOptions(const VariableDiffusionOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    if (options.diffusion < 0.0F || options.diffusion > 1.0F)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/diffusion", "diffusion must be in [0, 1]");
    }
}

void ValidateOptions(const CascadedFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    const bool size_valid = options.matrix_size > 0U;
    if (!size_valid)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/matrix_size", "matrix size must be positive");
    }
    const ScalarMatrixType type = GetMatrixType(options.generator);
    const bool type_valid = IsSupportedScalarType(type);
    if (!type_valid)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, path + "/generator", "matrix type is unsupported");
    }
    else if (size_valid && !IsValidScalarDimension(options.matrix_size, type))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/matrix_size",
                 "matrix size is unsupported by the selected matrix type");
    }
    std::visit(overloaded{[](ScalarMatrixType) {},
                          [&](const VariableDiffusionOptions& generator) {
                              ValidateOptions(generator, path + "/generator/VariableDiffusionOptions", issues);
                          }},
               options.generator);

    if (options.stage_count == std::numeric_limits<uint32_t>::max())
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, path + "/stage_count",
                 "stage count plus the initial matrix exceeds uint32_t capacity");
    }

    if (options.sparsity < 1.0F)
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/sparsity", "sparsity must be at least one");
    }

    if (size_valid && CascadedNegativeGainHasFractionalExponents(options.matrix_size, options.stage_count,
                                                                 options.sparsity, options.gain_per_samples))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/gain_per_samples",
                 "negative gain is unsupported with fractional delay exponents");
    }

    if (size_valid && options.sparsity >= 1.0F &&
        !CascadedShiftFits(options.matrix_size, options.stage_count, options.sparsity))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, path + "/stage_count",
                 "generated delay shifts plus block padding exceed uint32_t capacity");
    }

    if (size_valid && options.sparsity >= 1.0F &&
        CascadedGainCanOverflow(options.matrix_size, options.stage_count, options.sparsity, options.gain_per_samples))
    {
        AddIssue(issues, ConfigErrorCode::CapacityOverflow, path + "/gain_per_samples",
                 "generated stage gains exceed float capacity");
    }
}

void ValidateOptions(const TimeVaryingFeedbackMatrixOptions& options, const std::string& path,
                     std::vector<ConfigIssue>& issues)
{
    const bool mode_valid = IsSupportedTimeVaryingMode(options.mode);
    if (!mode_valid)
    {
        AddIssue(issues, ConfigErrorCode::UnsupportedValue, path + "/mode", "time-varying matrix mode is unsupported");
    }

    const bool size_valid = options.matrix_size >= 2U && (options.matrix_size % 2U) == 0U;
    if (!size_valid || (options.mode == TimeVaryingMatrixMode::Hadamard && !std::has_single_bit(options.matrix_size)))
    {
        AddIssue(issues, ConfigErrorCode::InvalidValue, path + "/matrix_size",
                 "matrix size must be even and at least two; Hadamard mode requires a power of two");
    }
    if (size_valid && mode_valid && !options.time_varying_config.empty() &&
        options.time_varying_config.size() != options.matrix_size / 2U)
    {
        AddIssue(issues, ConfigErrorCode::SizeMismatch, path + "/time_varying_config",
                 "modulation count must equal the rotation block count");
    }

    ValidateModulationOptions(options.time_varying_config, path + "/time_varying_config", issues);
}
} // namespace sfFDN::detail
