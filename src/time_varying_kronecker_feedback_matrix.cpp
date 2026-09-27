// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "sffdn/time_varying_kronecker_feedback_matrix.h"

#include "processor_option_validation.h"
#include "sincos.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace
{

double NormalizeIncrement(float frequency)
{
    const auto value = static_cast<double>(frequency);
    return value - std::floor(value);
}

void ValidateModulation(const sfFDN::ModulationOptions& modulation)
{
    if (modulation.frequency < 0.0F)
    {
        throw std::invalid_argument("TimeVaryingKroneckerFeedbackMatrix: modulation frequency must be non-negative");
    }
    if (!(std::abs(modulation.amplitude) <= 1.0F))
    {
        throw std::invalid_argument("TimeVaryingKroneckerFeedbackMatrix: modulation amplitude must be in [-1, 1]");
    }
    if (modulation.initial_phase < 0.0F || modulation.initial_phase > 1.0F)
    {
        throw std::invalid_argument("TimeVaryingKroneckerFeedbackMatrix: modulation initial phase must be in [0, 1]");
    }
}

float AngleOffsetAt(const sfFDN::ModulationOptions& modulation, double phase) noexcept SFFDN_NONBLOCKING
{
    double offset_phase = phase + static_cast<double>(modulation.initial_phase);
    offset_phase -= std::floor(offset_phase);
    return std::numbers::pi_v<float> * modulation.amplitude * sfFDN::SineTableLookup(static_cast<float>(offset_phase));
}

} // namespace

namespace sfFDN
{

TimeVaryingKroneckerFeedbackMatrix::TimeVaryingKroneckerFeedbackMatrix(
    const TimeVaryingKroneckerFeedbackMatrixOptions& options)
    : options_(detail::RequireValidOptions(options))
    , matrix_(options_.matrix)
    , modulation_configs_(matrix_.StageCount())
    , phase_increments_(matrix_.StageCount(), 0.0)
    , phases_(matrix_.StageCount(), 0.0)
    , angle_offsets_(static_cast<size_t>(matrix_.StageCount()) * kAngleChunkSize, 0.0F)
{
    if (!options_.time_varying_config.empty())
    {
        SetTimeVaryingConfig(options_.time_varying_config);
    }
}

void TimeVaryingKroneckerFeedbackMatrix::SetAngles(std::span<const float> radians)
{
    if (radians.size() != matrix_.StageCount())
    {
        throw std::invalid_argument("TimeVaryingKroneckerFeedbackMatrix: expected one angle per stage");
    }
    std::vector<float> candidate(radians.begin(), radians.end());
    matrix_.SetAngles(candidate);
    options_.matrix.angles = std::move(candidate);
}

void TimeVaryingKroneckerFeedbackMatrix::SetTimeVaryingConfig(std::span<const ModulationOptions> modulation_configs)
{
    if (!modulation_configs.empty() && modulation_configs.size() != matrix_.StageCount())
    {
        throw std::invalid_argument(
            "TimeVaryingKroneckerFeedbackMatrix: expected one modulation configuration per stage");
    }

    std::vector<ModulationOptions> candidate(matrix_.StageCount());
    std::vector<double> increments(matrix_.StageCount(), 0.0);
    if (!modulation_configs.empty())
    {
        for (size_t stage = 0; stage < modulation_configs.size(); ++stage)
        {
            ValidateModulation(modulation_configs[stage]);
            candidate[stage] = modulation_configs[stage];
            increments[stage] = NormalizeIncrement(modulation_configs[stage].frequency);
        }
    }

    modulation_configs_ = std::move(candidate);
    phase_increments_ = std::move(increments);
    options_.time_varying_config.assign(modulation_configs.begin(), modulation_configs.end());
}

void TimeVaryingKroneckerFeedbackMatrix::Process(const AudioBuffer& input,
                                                 AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == matrix_.InputChannelCount());
    assert(output.ChannelCount() == matrix_.OutputChannelCount());
    assert(input.SampleCount() == output.SampleCount());

    const size_t sample_count = input.SampleCount();
    for (size_t block_start = 0; block_start < sample_count; block_start += kAngleChunkSize)
    {
        const size_t block_size = std::min(kAngleChunkSize, sample_count - block_start);
        uint32_t varying_stage_mask = 0U;
        for (uint32_t stage = 0; stage < matrix_.StageCount(); ++stage)
        {
            const size_t row = static_cast<size_t>(stage) * block_size;
            const double increment = phase_increments_[stage];
            const float amplitude = modulation_configs_[stage].amplitude;
            double phase = phases_[stage];

            if (increment == 0.0)
            {
                angle_offsets_[row] = AngleOffsetAt(modulation_configs_[stage], phase);
                continue;
            }
            if (amplitude == 0.0F)
            {
                angle_offsets_[row] = 0.0F;
                for (size_t sample = 0; sample < block_size; ++sample)
                {
                    phase += increment;
                    if (phase >= 1.0)
                    {
                        phase -= 1.0;
                    }
                }
                phases_[stage] = phase;
                continue;
            }

            varying_stage_mask |= 1U << stage;
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                angle_offsets_[row + sample] = AngleOffsetAt(modulation_configs_[stage], phase);
                phase += increment;
                if (phase >= 1.0)
                {
                    phase -= 1.0;
                }
            }
            phases_[stage] = phase;
        }

        const auto input_block = input.Offset(static_cast<uint32_t>(block_start), static_cast<uint32_t>(block_size));
        auto output_block = output.Offset(static_cast<uint32_t>(block_start), static_cast<uint32_t>(block_size));
        matrix_.ProcessWithAngleOffsets(
            input_block, output_block,
            std::span<const float>(angle_offsets_).first(static_cast<size_t>(matrix_.StageCount()) * block_size),
            varying_stage_mask);
    }
}

uint32_t TimeVaryingKroneckerFeedbackMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return matrix_.InputChannelCount();
}

uint32_t TimeVaryingKroneckerFeedbackMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return matrix_.OutputChannelCount();
}

uint32_t TimeVaryingKroneckerFeedbackMatrix::StageCount() const noexcept SFFDN_NONBLOCKING
{
    return matrix_.StageCount();
}

bool TimeVaryingKroneckerFeedbackMatrix::GetMatrix(std::span<float> matrix, uint64_t sample_index) const
{
    const uint32_t order = matrix_.InputChannelCount();
    const size_t element_count = static_cast<size_t>(order) * order;
    if (matrix.size() != element_count)
    {
        return false;
    }

    std::vector<float> offsets(static_cast<size_t>(matrix_.StageCount()) * order, 0.0F);
    for (uint32_t stage = 0; stage < matrix_.StageCount(); ++stage)
    {
        const double advanced = static_cast<double>(sample_index) * phase_increments_[stage];
        const double phase = advanced - std::floor(advanced);
        offsets[static_cast<size_t>(stage) * order] = AngleOffsetAt(modulation_configs_[stage], phase);
    }

    std::vector<float> basis(element_count, 0.0F);
    std::vector<float> transformed(element_count, 0.0F);
    AudioBuffer basis_buffer(order, order, basis);
    AudioBuffer transformed_buffer(order, order, transformed);
    for (uint32_t channel = 0; channel < order; ++channel)
    {
        basis_buffer.GetChannelSpan(channel)[channel] = 1.0F;
    }
    matrix_.ProcessWithAngleOffsets(basis_buffer, transformed_buffer, offsets, 0U);

    for (uint32_t row = 0; row < order; ++row)
    {
        const auto row_data = transformed_buffer.GetChannelSpan(row);
        for (uint32_t column = 0; column < order; ++column)
        {
            matrix[(static_cast<size_t>(row) * order) + column] = row_data[column];
        }
    }
    return true;
}

void TimeVaryingKroneckerFeedbackMatrix::Clear()
{
    std::ranges::fill(phases_, 0.0);
}

std::unique_ptr<AudioProcessor> TimeVaryingKroneckerFeedbackMatrix::Clone() const
{
    auto clone = std::make_unique<TimeVaryingKroneckerFeedbackMatrix>(options_);
    clone->phases_ = phases_;
    return clone;
}

} // namespace sfFDN
