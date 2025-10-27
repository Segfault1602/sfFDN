#include "sffdn/fdn.h"

#include "array_math.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_utils.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/parallel_gains.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>

namespace
{
constexpr uint32_t kDefaultBlockSize = 64;
}

namespace sfFDN
{
FDN::FDN(uint32_t order, uint32_t block_size, bool transpose)
    : delay_bank_(GetDelayLengths(order, block_size + 1, block_size * 10, DelayLengthType::Random), block_size)
    , filter_bank_(nullptr)
    , mixing_matrix_(std::make_unique<ScalarFeedbackMatrix>(order))
    , order_(order)
    , block_size_(block_size == 0 ? kDefaultBlockSize : block_size)
    , direct_gain_(1.f)
    , feedback_(order * block_size_, 0.f)
    , temp_buffer_(order * block_size_, 0.f)
    , tc_filter_(nullptr)
    , transpose_(transpose)
{
    input_gains_ = std::make_unique<ParallelGains>(order, ParallelGainsMode::Split, 0.5f);
    output_gains_ = std::make_unique<ParallelGains>(order, ParallelGainsMode::Merge, 0.5f);
    delay_bank_.SetDelays(std::vector<uint32_t>(order, 500), block_size_);
}

FDN::FDN(FDN&& other) noexcept
    : delay_bank_(std::move(other.delay_bank_))
    , filter_bank_(std::move(other.filter_bank_))
    , mixing_matrix_(std::move(other.mixing_matrix_))
    , input_gains_(std::move(other.input_gains_))
    , output_gains_(std::move(other.output_gains_))
    , order_(other.order_)
    , block_size_(other.block_size_)
    , direct_gain_(other.direct_gain_)
    , feedback_(std::move(other.feedback_))
    , temp_buffer_(std::move(other.temp_buffer_))
    , tc_filter_(std::move(other.tc_filter_))
    , transpose_(other.transpose_)
{
}

FDN& FDN::operator=(FDN&& other) noexcept
{
    if (this != &other)
    {
        delay_bank_ = std::move(other.delay_bank_);
        filter_bank_ = std::move(other.filter_bank_);
        mixing_matrix_ = std::move(other.mixing_matrix_);
        input_gains_ = std::move(other.input_gains_);
        output_gains_ = std::move(other.output_gains_);
        order_ = other.order_;
        block_size_ = other.block_size_;
        direct_gain_ = other.direct_gain_;
        feedback_ = std::move(other.feedback_);
        temp_buffer_ = std::move(other.temp_buffer_);
        tc_filter_ = std::move(other.tc_filter_);
        transpose_ = other.transpose_;
    }
    return *this;
}

void FDN::SetOrder(uint32_t order)
{
    if (order < 4)
    {
        std::println(std::cerr, "FDN must have at least 4 channels.");
        return;
    }

    if (order == order_)
    {
        return; // No change in size
    }

    order_ = order;
    feedback_.resize(order * block_size_, 0.f);
    temp_buffer_.resize(order * block_size_, 0.f);

    delay_bank_.SetDelays(std::vector<uint32_t>(order, 500), block_size_);
    filter_bank_ = nullptr;

    SetFeedbackMatrix(std::make_unique<ScalarFeedbackMatrix>(order));
    SetInputGains(std::make_unique<ParallelGains>(ParallelGainsMode::Split, std::vector<float>(order, 0.5f)));
    SetOutputGains(std::make_unique<ParallelGains>(ParallelGainsMode::Merge, std::vector<float>(order, 0.5f)));

    // tc_filter is always one channel so it is not impacted
    assert(tc_filter_ == nullptr || tc_filter_->InputChannelCount() == 1);
}

uint32_t FDN::GetOrder() const
{
    return order_;
}

void FDN::SetTranspose(bool transpose)
{
    transpose_ = transpose;
}

bool FDN::GetTranspose() const
{
    return transpose_;
}

bool FDN::SetInputGains(std::unique_ptr<AudioProcessor> gains)
{
    if (gains->InputChannelCount() != 1 || gains->OutputChannelCount() != order_)
    {
        std::println(std::cerr, "Input gains must have 1 input and {} output channels.", order_);
        assert(false);
        return false;
    }

    input_gains_ = std::move(gains);
    return true;
}

bool FDN::SetInputGains(std::span<const float> gains)
{
    if (gains.size() != order_)
    {
        std::println(std::cerr, "Input gains must have {} elements.", order_);
        assert(false);
        return false;
    }
    input_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::Split, gains);
    return true;
}

AudioProcessor* FDN::GetInputGains() const
{
    return input_gains_.get();
}

bool FDN::SetOutputGains(std::unique_ptr<AudioProcessor> gains)
{
    if (gains->InputChannelCount() != order_ || gains->OutputChannelCount() != 1)
    {
        std::println(std::cerr, "Output gains must have {} input and 1 output channels.", order_);
        return false;
    }

    output_gains_ = std::move(gains);
    return true;
}

bool FDN::SetOutputGains(std::span<const float> gains)
{
    if (gains.size() != order_)
    {
        std::println(std::cerr, "Output gains must have {} elements.", order_);
        return false;
    }
    output_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::Merge, gains);
    return true;
}

AudioProcessor* FDN::GetOutputGains() const
{
    return output_gains_.get();
}

void FDN::SetDirectGain(float gain)
{
    direct_gain_ = gain;
}

bool FDN::SetFilterBank(std::unique_ptr<AudioProcessor> filter_bank)
{
    if (filter_bank == nullptr)
    {
        filter_bank_ = nullptr;
        return true;
    }

    if (filter_bank->InputChannelCount() != order_ || filter_bank->OutputChannelCount() != order_)
    {
        std::println(std::cerr, "Filter bank must have {} input and output channels.", order_);
        return false;
    }

    filter_bank_ = std::move(filter_bank);
    return true;
}

AudioProcessor* FDN::GetFilterBank() const
{
    return filter_bank_.get();
}

bool FDN::SetDelays(const std::span<const uint32_t> delays)
{
    for (const auto& delay : delays)
    {
        if (delay == 0)
        {
            std::println(std::cerr, "Delay cannot be zero.");
            return false;
        }

        if (delay < block_size_)
        {
            std::println(std::cerr, "Delay {} is smaller than block size {}.", delay, block_size_);
            return false;
        }
    }

    if (delays.size() != order_)
    {
        std::println(std::cerr, "Delays must have {} elements.", order_);
        return false;
    }

    delay_bank_.SetDelays(delays, block_size_);
    return true;
}

const DelayBank& FDN::GetDelayBank() const
{
    return delay_bank_;
}

bool FDN::SetFeedbackMatrix(std::unique_ptr<AudioProcessor> mixing_matrix)
{
    if (mixing_matrix == nullptr)
    {
        std::println(std::cerr, "Feedback matrix cannot be null.");
        return false;
    }

    if (mixing_matrix->InputChannelCount() != order_ || mixing_matrix->OutputChannelCount() != order_)
    {
        std::println(std::cerr, "Feedback matrix must have {} input and output channels.", order_);
        return false;
    }

    mixing_matrix_ = std::move(mixing_matrix);
    return true;
}

AudioProcessor* FDN::GetFeedbackMatrix() const
{
    return mixing_matrix_.get();
}

bool FDN::SetTCFilter(std::unique_ptr<AudioProcessor> filter)
{
    if (filter == nullptr)
    {
        tc_filter_ = nullptr;
        return true;
    }

    if (filter->InputChannelCount() != 1 || filter->OutputChannelCount() != 1)
    {
        std::println(std::cerr, "TC filter must have 1 input and 1 output channel.");
        return false;
    }

    tc_filter_ = std::move(filter);
    return true;
}

AudioProcessor* FDN::GetTCFilter() const
{
    return tc_filter_.get();
}

void FDN::Process(const AudioBuffer& input, AudioBuffer& output) noexcept [[clang::nonblocking]]
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(input_gains_ != nullptr);
    assert(output_gains_ != nullptr);

    AudioBuffer mono_output = output.GetChannelBuffer(0);

    if (transpose_)
    {
        TickTranspose(input, mono_output);
    }
    else
    {
        Tick(input, mono_output);
    }

    if (output.ChannelCount() > 1)
    {
        // If output has more than one channel, copy the mono output to all channels
        for (uint32_t i = 1; i < output.ChannelCount(); ++i)
        {
            std::copy(mono_output.GetChannelSpan(0).begin(), mono_output.GetChannelSpan(0).end(),
                      output.GetChannelSpan(i).begin());
        }
    }
}

void FDN::TickInternal(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() * input.ChannelCount() <= temp_buffer_.size());

    const uint32_t block_size = input.SampleCount();

    AudioBuffer temp_buffer(block_size, order_, temp_buffer_);
    AudioBuffer feedback_buffer(block_size, order_, feedback_);

    if (filter_bank_)
    {
        delay_bank_.GetNextOutputs(temp_buffer);
        filter_bank_->Process(temp_buffer, feedback_buffer);
    }
    else
    {
        delay_bank_.GetNextOutputs(feedback_buffer);
    }

    output_gains_->Process(feedback_buffer, output);

    mixing_matrix_->Process(feedback_buffer, temp_buffer);

    input_gains_->Process(input, feedback_buffer);
    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    delay_bank_.AddNextInputs(feedback_buffer);

    if (tc_filter_)
    {
        tc_filter_->Process(output, output);
    }

    ArrayMath::ScaleAccumulate(input.GetChannelSpan(0), direct_gain_, output.GetChannelSpan(0));
}

void FDN::TickTransposeInternal(const AudioBuffer& input, AudioBuffer& output)
{
    const uint32_t block_size = input.SampleCount();

    AudioBuffer temp_buffer(block_size, order_, temp_buffer_);
    AudioBuffer feedback_buffer(block_size, order_, feedback_);

    input_gains_->Process(input, temp_buffer);

    delay_bank_.GetNextOutputs(feedback_buffer);

    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    mixing_matrix_->Process(feedback_buffer, temp_buffer);

    if (filter_bank_)
    {
        filter_bank_->Process(temp_buffer, feedback_buffer);
        std::swap(feedback_buffer, temp_buffer);
    }

    delay_bank_.AddNextInputs(temp_buffer);

    output_gains_->Process(temp_buffer, output);
    if (tc_filter_)
    {
        tc_filter_->Process(output, output);
    }

    ArrayMath::ScaleAccumulate(input.GetChannelSpan(0), direct_gain_, output.GetChannelSpan(0));
}

void FDN::Tick(const AudioBuffer& input, AudioBuffer& output)
{
    const uint32_t block_count = input.SampleCount() / block_size_;

    for (auto i = 0u; i < block_count; ++i)
    {
        AudioBuffer input_block = input.Offset(i * block_size_, block_size_);
        AudioBuffer output_block = output.Offset(i * block_size_, block_size_);

        TickInternal(input_block, output_block);
    }

    uint32_t remaining_samples = input.SampleCount() % block_size_;
    assert(block_size_ * block_count + remaining_samples == input.SampleCount());

    if (remaining_samples > 0)
    {
        AudioBuffer input_block = input.Offset(block_count * block_size_, remaining_samples);
        AudioBuffer output_block = output.Offset(block_count * block_size_, remaining_samples);

        TickInternal(input_block, output_block);
    }
}

void FDN::TickTranspose(const AudioBuffer& input, AudioBuffer& output)
{
    const uint32_t block_count = input.SampleCount() / block_size_;

    for (auto i = 0u; i < block_count; ++i)
    {
        AudioBuffer input_block = input.Offset(i * block_size_, block_size_);
        AudioBuffer output_block = output.Offset(i * block_size_, block_size_);

        TickTransposeInternal(input_block, output_block);
    }

    uint32_t remaining_samples = input.SampleCount() % block_size_;
    assert(block_size_ * block_count + remaining_samples == input.SampleCount());

    if (remaining_samples > 0)
    {
        AudioBuffer input_block = input.Offset(block_count * block_size_, remaining_samples);
        AudioBuffer output_block = output.Offset(block_count * block_size_, remaining_samples);

        TickTransposeInternal(input_block, output_block);
    }
}

void FDN::Clear()
{
    delay_bank_.Clear();
    if (filter_bank_)
    {
        filter_bank_->Clear();
    }
    if (mixing_matrix_)
    {
        mixing_matrix_->Clear();
    }
    if (input_gains_)
    {
        input_gains_->Clear();
    }
    if (output_gains_)
    {
        output_gains_->Clear();
    }
    if (tc_filter_)
    {
        tc_filter_->Clear();
    }

    std::ranges::fill(feedback_, 0.f);
    std::ranges::fill(temp_buffer_, 0.f);
}

std::unique_ptr<AudioProcessor> FDN::Clone() const
{
    return CloneFDN();
}

std::unique_ptr<FDN> FDN::CloneFDN() const
{
    auto clone = std::make_unique<FDN>(order_, block_size_, transpose_);

    assert(input_gains_ != nullptr);
    clone->SetInputGains(input_gains_->Clone());

    assert(output_gains_ != nullptr);
    clone->SetOutputGains(output_gains_->Clone());

    clone->SetFilterBank(filter_bank_ ? filter_bank_->Clone() : nullptr);
    clone->SetDelays(delay_bank_.GetDelays());
    clone->SetFeedbackMatrix(mixing_matrix_ ? mixing_matrix_->Clone() : nullptr);
    clone->SetTCFilter(tc_filter_ ? tc_filter_->Clone() : nullptr);
    clone->SetDirectGain(direct_gain_);

    assert(clone->order_ == order_);
    assert(clone->block_size_ == block_size_);
    assert(clone->transpose_ == transpose_);
    assert(clone->direct_gain_ == direct_gain_);
    assert(clone->InputChannelCount() == InputChannelCount());
    assert(clone->OutputChannelCount() == OutputChannelCount());

    return clone;
}

} // namespace sfFDN