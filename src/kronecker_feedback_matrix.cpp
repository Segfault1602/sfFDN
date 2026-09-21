// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include "sffdn/kronecker_feedback_matrix.h"

#include "processor_option_validation.h"
#include "sincos.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace
{

constexpr size_t kCoefficientChunkSize = 128U;

uint32_t GetStageCount(uint32_t order)
{
    return std::bit_width(order) - 1U;
}

float WrapAngle(float radians) noexcept SFFDN_NONBLOCKING
{
    constexpr float kTwoPi = 2.0F * std::numbers::pi_v<float>;
    return radians - (kTwoPi * std::floor((radians + std::numbers::pi_v<float>) / kTwoPi));
}

void ApplyKernel(sfFDN::KroneckerKernelType type, float sine, float cosine, float first, float second, float& first_out,
                 float& second_out) noexcept SFFDN_NONBLOCKING
{
    if (type == sfFDN::KroneckerKernelType::Rotation)
    {
        first_out = (cosine * first) - (sine * second);
        second_out = (sine * first) + (cosine * second);
        return;
    }

    first_out = (cosine * first) + (sine * second);
    second_out = (sine * first) - (cosine * second);
}

bool IsSameView(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output, uint32_t order) noexcept SFFDN_NONBLOCKING
{
    for (uint32_t channel = 0; channel < order; ++channel)
    {
        const auto input_channel = input.GetChannelSpan(channel);
        const auto output_channel = output.GetChannelSpan(channel);
        if (input_channel.data() != output_channel.data() || input_channel.size() != output_channel.size())
        {
            return false;
        }
    }
    return true;
}

void CopyInput(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output, uint32_t order) noexcept SFFDN_NONBLOCKING
{
    const size_t sample_count = input.SampleCount();
    for (uint32_t channel = 0; channel < order; ++channel)
    {
        const auto input_channel = input.GetChannelSpan(channel);
        const auto output_channel = output.GetChannelSpan(channel);
        for (size_t sample = 0; sample < sample_count; ++sample)
        {
            output_channel[sample] = input_channel[sample];
        }
    }
}

void ApplyConstantStage(sfFDN::AudioBuffer& output, uint32_t order, uint32_t stage, sfFDN::KroneckerKernelType type,
                        float sine, float cosine, size_t block_start, size_t block_size) noexcept SFFDN_NONBLOCKING
{
    const uint32_t stride = 1U << stage;
    const uint32_t group_size = 2U * stride;
    for (uint32_t group = 0; group < order; group += group_size)
    {
        for (uint32_t offset = 0; offset < stride; ++offset)
        {
            const auto first_channel = output.GetChannelSpan(group + offset);
            const auto second_channel = output.GetChannelSpan(group + stride + offset);
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                const size_t index = block_start + sample;
                const float first = first_channel[index];
                const float second = second_channel[index];
                ApplyKernel(type, sine, cosine, first, second, first_channel[index], second_channel[index]);
            }
        }
    }
}

void ApplyVaryingStage(sfFDN::AudioBuffer& output, uint32_t order, uint32_t stage, sfFDN::KroneckerKernelType type,
                       std::span<const float> sines, std::span<const float> cosines, size_t block_start,
                       size_t block_size) noexcept SFFDN_NONBLOCKING
{
    const uint32_t stride = 1U << stage;
    const uint32_t group_size = 2U * stride;
    for (uint32_t group = 0; group < order; group += group_size)
    {
        for (uint32_t offset = 0; offset < stride; ++offset)
        {
            const auto first_channel = output.GetChannelSpan(group + offset);
            const auto second_channel = output.GetChannelSpan(group + stride + offset);
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                const size_t index = block_start + sample;
                const float first = first_channel[index];
                const float second = second_channel[index];
                ApplyKernel(type, sines[sample], cosines[sample], first, second, first_channel[index],
                            second_channel[index]);
            }
        }
    }
}

} // namespace

namespace sfFDN
{

KroneckerFeedbackMatrix::KroneckerFeedbackMatrix(const KroneckerFeedbackMatrixOptions& options)
    : order_(detail::RequireValidOptions(options).matrix_size)
    , stage_count_(GetStageCount(order_))
    , angles_(stage_count_, std::numbers::pi_v<float> / 4.0F)
    , kernel_types_(stage_count_, KroneckerKernelType::Rotation)
    , sines_(stage_count_)
    , cosines_(stage_count_)
{
    if (!options.kernel_types.empty())
    {
        kernel_types_ = options.kernel_types;
    }
    if (!options.angles.empty())
    {
        SetAngles(options.angles);
    }
    else
    {
        SetAngles(angles_);
    }
}

void KroneckerFeedbackMatrix::SetAngles(std::span<const float> radians)
{
    if (radians.size() != stage_count_)
    {
        throw std::invalid_argument("KroneckerFeedbackMatrix: expected one angle per stage");
    }

    for (size_t stage = 0; stage < radians.size(); ++stage)
    {
        angles_[stage] = WrapAngle(radians[stage]);
        SinCosUnit(angles_[stage], sines_[stage], cosines_[stage]);
    }
}

void KroneckerFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == order_);
    assert(output.ChannelCount() == order_);
    assert(input.SampleCount() == output.SampleCount());

    if (!IsSameView(input, output, order_))
    {
        CopyInput(input, output, order_);
    }
    for (uint32_t stage = 0; stage < stage_count_; ++stage)
    {
        ApplyConstantStage(output, order_, stage, kernel_types_[stage], sines_[stage], cosines_[stage], 0U,
                           output.SampleCount());
    }
}

void KroneckerFeedbackMatrix::ProcessWithAngleOffsets(const AudioBuffer& input, AudioBuffer& output,
                                                      std::span<const float> angle_offsets,
                                                      uint32_t varying_stage_mask) const noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == order_);
    assert(output.ChannelCount() == order_);
    assert(input.SampleCount() == output.SampleCount());
    const size_t sample_count = input.SampleCount();
    assert(angle_offsets.size() == static_cast<size_t>(stage_count_) * sample_count);
    assert((varying_stage_mask >> stage_count_) == 0U);
    if (sample_count == 0U)
    {
        return;
    }

    if (!IsSameView(input, output, order_))
    {
        CopyInput(input, output, order_);
    }

    std::array<float, kCoefficientChunkSize> varying_sines{};
    std::array<float, kCoefficientChunkSize> varying_cosines{};
    for (size_t block_start = 0; block_start < sample_count; block_start += kCoefficientChunkSize)
    {
        const size_t block_size = std::min(kCoefficientChunkSize, sample_count - block_start);
        for (uint32_t stage = 0; stage < stage_count_; ++stage)
        {
            const size_t angle_row = static_cast<size_t>(stage) * sample_count;
            if ((varying_stage_mask & (1U << stage)) == 0U)
            {
                const float offset = angle_offsets[angle_row];
                if (offset == 0.0F)
                {
                    ApplyConstantStage(output, order_, stage, kernel_types_[stage], sines_[stage], cosines_[stage],
                                       block_start, block_size);
                }
                else
                {
                    float sine = 0.0F;
                    float cosine = 0.0F;
                    SinCosUnit(angles_[stage] + offset, sine, cosine);
                    ApplyConstantStage(output, order_, stage, kernel_types_[stage], sine, cosine, block_start,
                                       block_size);
                }
                continue;
            }

            for (size_t sample = 0; sample < block_size; ++sample)
            {
                SinCosUnit(angles_[stage] + angle_offsets[angle_row + block_start + sample], varying_sines[sample],
                           varying_cosines[sample]);
            }
            ApplyVaryingStage(output, order_, stage, kernel_types_[stage],
                              std::span<const float>(varying_sines).first(block_size),
                              std::span<const float>(varying_cosines).first(block_size), block_start, block_size);
        }
    }
}

uint32_t KroneckerFeedbackMatrix::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

uint32_t KroneckerFeedbackMatrix::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return order_;
}

uint32_t KroneckerFeedbackMatrix::StageCount() const noexcept SFFDN_NONBLOCKING
{
    return stage_count_;
}

bool KroneckerFeedbackMatrix::GetMatrix(std::span<float> matrix) const
{
    const size_t element_count = static_cast<size_t>(order_) * order_;
    if (matrix.size() != element_count)
    {
        return false;
    }

    std::vector<float> column(order_, 0.0F);
    for (uint32_t input_channel = 0; input_channel < order_; ++input_channel)
    {
        std::ranges::fill(column, 0.0F);
        column[input_channel] = 1.0F;
        for (uint32_t stage = 0; stage < stage_count_; ++stage)
        {
            const uint32_t stride = 1U << stage;
            const uint32_t group_size = 2U * stride;
            for (uint32_t group = 0; group < order_; group += group_size)
            {
                for (uint32_t offset = 0; offset < stride; ++offset)
                {
                    const uint32_t first_index = group + offset;
                    const uint32_t second_index = group + stride + offset;
                    const float first = column[first_index];
                    const float second = column[second_index];
                    ApplyKernel(kernel_types_[stage], sines_[stage], cosines_[stage], first, second,
                                column[first_index], column[second_index]);
                }
            }
        }

        for (uint32_t output_channel = 0; output_channel < order_; ++output_channel)
        {
            matrix[(static_cast<size_t>(output_channel) * order_) + input_channel] = column[output_channel];
        }
    }
    return true;
}

void KroneckerFeedbackMatrix::Clear()
{
}

std::unique_ptr<AudioProcessor> KroneckerFeedbackMatrix::Clone() const
{
    return std::make_unique<KroneckerFeedbackMatrix>(KroneckerFeedbackMatrixOptions{
        .matrix_size = order_,
        .angles = angles_,
        .kernel_types = kernel_types_,
    });
}

} // namespace sfFDN
