#include "sffdn/fdn.h"

#include <cassert>
#include <iostream>
#include <print>

#include "array_math.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/parallel_gains.h"

namespace sfFDN
{
FDN::FDN(uint32_t N, uint32_t block_size, bool transpose)
    : filter_bank_(nullptr)
    , mixing_matrix_(std::make_unique<ScalarFeedbackMatrix>(N))
    , N_(N)
    , block_size_(block_size)
    , direct_gain_(1.f)
    , feedback_(N * block_size, 0.f)
    , temp_buffer_(N * block_size, 0.f)
    , tc_filter_(nullptr)
    , transpose_(transpose)
{
    input_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::Multiplexed);
    output_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::DeMultiplexed);
}

bool FDN::SetInputGains(std::unique_ptr<AudioProcessor> gains)
{
    if (gains->InputChannelCount() != 1 || gains->OutputChannelCount() != N_)
    {
        std::println(std::cerr, "Input gains must have 1 input and {} output channels.", N_);
        return false;
    }

    input_gains_ = std::move(gains);
    return true;
}

bool FDN::SetInputGains(std::span<const float> gains)
{
    if (gains.size() != N_)
    {
        std::println(std::cerr, "Input gains must have {} elements.", N_);
        return false;
    }
    input_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::Multiplexed, gains);
    return true;
}

bool FDN::SetOutputGains(std::unique_ptr<AudioProcessor> gains)
{
    if (gains->InputChannelCount() != N_ || gains->OutputChannelCount() != 1)
    {
        std::println(std::cerr, "Output gains must have {} input and 1 output channels.", N_);
        return false;
    }

    output_gains_ = std::move(gains);
    return true;
}

bool FDN::SetOutputGains(std::span<const float> gains)
{
    if (gains.size() != N_)
    {
        std::println(std::cerr, "Output gains must have {} elements.", N_);
        return false;
    }
    output_gains_ = std::make_unique<ParallelGains>(ParallelGainsMode::DeMultiplexed, gains);
    return true;
}

void FDN::SetDirectGain(float gain)
{
    direct_gain_ = gain;
}

void FDN::SetFilterBank(std::unique_ptr<AudioProcessor> filter_bank)
{
    if (filter_bank->InputChannelCount() != N_ || filter_bank->OutputChannelCount() != N_)
    {
        std::println(std::cerr, "Filter bank must have {} input and output channels.", N_);
        return;
    }
    filter_bank_ = std::move(filter_bank);
}

void FDN::SetDelays(const std::span<const uint32_t> delays)
{
    for (const auto& delay : delays)
    {
        if (delay == 0)
        {
            std::println(std::cerr, "Delay cannot be zero.");
            return;
        }
        if (delay < block_size_)
        {
            std::println(std::cerr, "Delay must be at least as long as the block size ({}).", block_size_);
            return;
        }
    }
    delay_bank_.SetDelays(delays, block_size_);
}

void FDN::SetFeedbackMatrix(std::unique_ptr<AudioProcessor> mixing_matrix)
{
    mixing_matrix_ = std::move(mixing_matrix);
}

void FDN::SetTCFilter(std::unique_ptr<AudioProcessor> filter)
{
    tc_filter_ = std::move(filter);
}

void FDN::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);

    assert(input.SampleCount() == block_size_);

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

void FDN::Tick(const AudioBuffer& input, AudioBuffer& output)
{
    AudioBuffer temp_buffer(block_size_, N_, temp_buffer_.data());
    AudioBuffer feeback_buffer(block_size_, N_, feedback_.data());

    delay_bank_.GetNextOutputs(feeback_buffer);
    if (filter_bank_)
    {
        filter_bank_->Process(feeback_buffer, feeback_buffer);
    }

    output_gains_->Process(feeback_buffer, output);

    mixing_matrix_->Process(feeback_buffer, temp_buffer);

    input_gains_->Process(input, feeback_buffer);
    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    delay_bank_.AddNextInputs(feeback_buffer);

    if (tc_filter_)
    {
        tc_filter_->Process(output, output);
    }

    ArrayMath::ScaleAccumulate(input.GetChannelSpan(0), direct_gain_, output.GetChannelSpan(0));
}

void FDN::TickTranspose(const AudioBuffer& input, AudioBuffer& output)
{
    AudioBuffer temp_buffer(block_size_, N_, temp_buffer_.data());
    AudioBuffer feeback_buffer(block_size_, N_, feedback_.data());

    input_gains_->Process(input, temp_buffer);

    delay_bank_.GetNextOutputs(feeback_buffer);

    ArrayMath::Add(feedback_, temp_buffer_, feedback_);

    mixing_matrix_->Process(feeback_buffer, temp_buffer);

    if (filter_bank_)
    {
        filter_bank_->Process(temp_buffer, temp_buffer);
    }

    delay_bank_.AddNextInputs(temp_buffer);

    output_gains_->Process(temp_buffer, output);
    if (tc_filter_)
    {
        tc_filter_->Process(output, output);
    }

    ArrayMath::ScaleAccumulate(input.GetChannelSpan(0), direct_gain_, output.GetChannelSpan(0));
}

} // namespace sfFDN