#include "fdn.h"

#include <Eigen/Core>
#include <arm_neon.h>
#include <cassert>
#include <iostream>

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
    , input_gain_(N, 1.f)
    , output_gain_(N, 1.f)
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
    assert(gains.size() == input_gain_.size());
    std::copy(gains.begin(), gains.end(), input_gain_.begin());
}

void FDN::SetOutputGains(const std::span<const float> gains)
{
    assert(gains.size() == output_gain_.size());
    std::copy(gains.begin(), gains.end(), output_gain_.begin());
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

    const size_t col = N_;
    const size_t row = block_size_;

    Eigen::Map<const Eigen::VectorXf> input_map(input.data(), row);
    Eigen::Map<Eigen::VectorXf> input_gain_map(input_gain_.data(), N_);

    ColMajorMatrix temp_buffer_map(temp_buffer_.data(), row, col);
    temp_buffer_map = input_map * input_gain_map.transpose();

    if (schroeder_section_)
    {
        schroeder_section_->ProcessBlock(temp_buffer_, temp_buffer_);
    }

    delay_bank_.GetNextOutputs(feedback_);
    if (!bypass_absorption_)
    {
        filter_bank_.Tick(feedback_, feedback_);
    }

    ColMajorMatrix feedback_map(feedback_.data(), row, col);

    Eigen::Map<Eigen::VectorXf> output_map(output.data(), output.size());
    Eigen::Map<Eigen::VectorXf> output_gain_map(output_gain_.data(), N_);

    output_map = feedback_map * output_gain_map;

    mixing_matrix_->Tick(feedback_, feedback_);

    feedback_map.noalias() += temp_buffer_map;
    delay_bank_.AddNextInputs(feedback_);

    if (tc_filter_)
    {
        tc_filter_->ProcessBlock(output.data(), output.data(), output.size());
    }

    output_map.noalias() += input_map * direct_gain_;
}

void FDN::TickTranspose(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());
    assert(input.size() == block_size_);

    const size_t col = N_;
    const size_t row = block_size_;

    Eigen::Map<const Eigen::VectorXf> input_map(input.data(), row);
    Eigen::Map<Eigen::VectorXf> input_gain_map(input_gain_.data(), N_);

    ColMajorMatrix temp_buffer_map(temp_buffer_.data(), row, col);
    temp_buffer_map = input_map * input_gain_map.transpose();

    if (schroeder_section_)
    {
        schroeder_section_->ProcessBlock(temp_buffer_, temp_buffer_);
    }

    delay_bank_.GetNextOutputs(feedback_);

    ColMajorMatrix feedback_map(feedback_.data(), row, col);
    feedback_map.noalias() += temp_buffer_map;

    mixing_matrix_->Tick(feedback_, feedback_);
    if (!bypass_absorption_)
    {
        filter_bank_.Tick(feedback_, feedback_);
    }

    delay_bank_.AddNextInputs(feedback_);

    Eigen::Map<Eigen::VectorXf> output_map(output.data(), output.size());
    Eigen::Map<Eigen::VectorXf> output_gain_map(output_gain_.data(), N_);

    output_map = feedback_map * output_gain_map;
    if (tc_filter_)
    {
        tc_filter_->ProcessBlock(output.data(), output.data(), output.size());
    }

    output_map.noalias() += input_map * direct_gain_;
}

} // namespace fdn