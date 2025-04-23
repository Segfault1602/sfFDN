#include "fdn.h"

#include <Eigen/Core>
#include <arm_neon.h>
#include <cassert>
#include <iostream>

#include "array_math.h"

namespace
{
using RowMajorConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using RowMajorMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ColMajorMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

} // namespace

namespace fdn
{
FDN::FDN(size_t N, size_t block_size, bool transpose)
    : delay_bank_(N)
    , filter_bank_(N)
    , mixing_matrix_(std::make_unique<MixMat>(N))
    , N_(N)
    , block_size_(block_size)
    , direct_gain_(1.f)
    , feedback_(N * block_size, 0.f)
    , temp_buffer_(N * block_size, 0.f)
    , tc_filter_(nullptr)
    , schroeder_section_(nullptr)
    , bypass_absorption_(true)
    , transpose_(transpose)
{
}

void FDN::Clear()
{
    delay_bank_.Clear();
    filter_bank_.Clear();
}

void FDN::SetInputGains(const std::span<const float> gains)
{
    assert(gains.size() == N_);
    input_gains_.SetGains(gains);
}

void FDN::SetOutputGains(const std::span<const float> gains)
{
    assert(gains.size() == N_);
    output_gains_.SetGains(gains);
}

void FDN::SetDirectGain(float gain)
{
    direct_gain_ = gain;
}

FilterBank* FDN::GetFilterBank()
{
    return &filter_bank_;
}

DelayBank* FDN::GetDelayBank()
{
    return &delay_bank_;
}

void FDN::SetFeedbackMatrix(std::unique_ptr<FeedbackMatrix> mixing_matrix)
{
    mixing_matrix_ = std::move(mixing_matrix);
}

FeedbackMatrix* FDN::GetMixingMatrix()
{
    return mixing_matrix_.get();
}

void FDN::SetBypassAbsorption(bool bypass)
{
    bypass_absorption_ = bypass;
}

void FDN::SetDelayModulation(float freq, float depth)
{
    delay_bank_.SetModulation(freq, depth);
}

void FDN::SetTCFilter(std::unique_ptr<Filter> filter)
{
    tc_filter_ = std::move(filter);
}

void FDN::SetSchroederSection(std::unique_ptr<SchroederAllpassSection> section)
{
    schroeder_section_ = std::move(section);
}

void FDN::Tick(const std::span<const float> input, std::span<float> output)
{
    if (transpose_)
    {
        TickTranspose(input, output);
        return;
    }

    assert(input.size() == output.size());
    assert(input.size() == block_size_);

    input_gains_.ProcessBlock(input, temp_buffer_);

    if (schroeder_section_)
    {
        schroeder_section_->ProcessBlock(temp_buffer_, temp_buffer_);
    }

    delay_bank_.GetNextOutputs(feedback_);
    if (!bypass_absorption_)
    {
        filter_bank_.Tick(feedback_, feedback_);
    }

    output_gains_.ProcessBlock(feedback_, output);

    mixing_matrix_->Tick(feedback_, feedback_);

    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    delay_bank_.AddNextInputs(feedback_);

    if (tc_filter_)
    {
        tc_filter_->ProcessBlock(output.data(), output.data(), output.size());
    }

    ArrayMath::ScaleAdd(input, direct_gain_, output, output);
}

void FDN::TickTranspose(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());
    assert(input.size() == block_size_);

    const size_t col = N_;
    const size_t row = block_size_;

    input_gains_.ProcessBlock(input, temp_buffer_);

    if (schroeder_section_)
    {
        schroeder_section_->ProcessBlock(temp_buffer_, temp_buffer_);
    }

    delay_bank_.GetNextOutputs(feedback_);

    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    mixing_matrix_->Tick(feedback_, feedback_);
    if (!bypass_absorption_)
    {
        filter_bank_.Tick(feedback_, feedback_);
    }

    delay_bank_.AddNextInputs(feedback_);

    output_gains_.ProcessBlock(feedback_, output);
    if (tc_filter_)
    {
        tc_filter_->ProcessBlock(output.data(), output.data(), output.size());
    }

    ArrayMath::ScaleAdd(input, direct_gain_, output, output);
}

} // namespace fdn