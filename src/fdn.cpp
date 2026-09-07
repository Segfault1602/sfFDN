#include "sffdn/fdn.h"

#include "array_math.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/channel_matrix.h"
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
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef HAVE_XMMINTRIN_H
#include <xmmintrin.h>
#endif

namespace
{

class ScopedNoDenormals
{
  public:
    ScopedNoDenormals()
#ifdef HAVE_XMMINTRIN_H
        : old_mxcsr_(_mm_getcsr())
#endif
    {
#ifdef HAVE_XMMINTRIN_H
        constexpr intptr_t kMask = 0x8040;
        _mm_setcsr(old_mxcsr_ | kMask); // Set DAZ and FTZ bits
#endif
    }

    ~ScopedNoDenormals()
    {
#ifdef HAVE_XMMINTRIN_H
        _mm_setcsr(old_mxcsr_); // Restore old MXCSR
#endif
    }

    ScopedNoDenormals(const ScopedNoDenormals&) = delete;
    ScopedNoDenormals& operator=(const ScopedNoDenormals&) = delete;
    ScopedNoDenormals(ScopedNoDenormals&&) = delete;
    ScopedNoDenormals& operator=(ScopedNoDenormals&&) = delete;

  private:
#ifdef HAVE_XMMINTRIN_H
    unsigned int old_mxcsr_{};
#endif
};
} // namespace

namespace sfFDN
{
namespace
{
void ValidateTopology(const FDNTopology& topology)
{
    // Block size needs to be stricly greater than 1
    if (topology.block_size < 1)
    {
        throw std::invalid_argument("Block size must be at least 1.");
    }

    if (topology.order == 0 || topology.input_channel_count == 0 || topology.output_channel_count == 0)
    {
        throw std::invalid_argument("FDN order and external channel counts must be greater than zero.");
    }
}

DelayBankOptions MakeDefaultDelayBankOptions(const FDNTopology& topology)
{
    ValidateTopology(topology);
    return DelayBankOptions{
        .delays =
            GetDelayLengths(topology.order, topology.block_size + 1, topology.block_size * 10, DelayLengthType::Random),
        .block_size = topology.block_size,
    };
}

std::unique_ptr<AudioProcessor> MakeDefaultInputRouting(uint32_t input_channel_count, uint32_t order)
{
    if (input_channel_count == 1U)
    {
        return std::make_unique<ParallelGains>(ParallelGainsMode::Split, std::vector<float>(order, 0.5f));
    }

    return std::make_unique<ChannelMatrix>(ChannelMatrixOptions{
        .input_channel_count = input_channel_count,
        .output_channel_count = order,
        .coefficients = std::vector<float>(static_cast<size_t>(order) * input_channel_count, 0.5f),
    });
}

std::unique_ptr<AudioProcessor> MakeDefaultOutputRouting(uint32_t order, uint32_t output_channel_count)
{
    if (output_channel_count == 1U)
    {
        return std::make_unique<ParallelGains>(ParallelGainsMode::Merge, std::vector<float>(order, 0.5f));
    }

    return std::make_unique<ChannelMatrix>(ChannelMatrixOptions{
        .input_channel_count = order,
        .output_channel_count = output_channel_count,
        .coefficients = std::vector<float>(static_cast<size_t>(output_channel_count) * order, 0.5f),
    });
}
} // namespace

FDN::FDN(const FDNTopology& topology)
    : delay_bank_(MakeDefaultDelayBankOptions(topology))
    , filter_bank_(nullptr)
    , mixing_matrix_(std::make_unique<ScalarFeedbackMatrix>(
          ScalarFeedbackMatrixOptions{.source = GeneratedMatrixOptions{.matrix_size = topology.order}}))
    , direct_path_(nullptr)
    , order_(topology.order)
    , block_size_(topology.block_size)
    , input_channel_count_(topology.input_channel_count)
    , output_channel_count_(topology.output_channel_count)
    , direct_gain_(topology.input_channel_count == topology.output_channel_count ? 1.f : 0.f)
    , feedback_(static_cast<size_t>(topology.order) * topology.block_size, 0.f)
    , temp_buffer_(static_cast<size_t>(topology.order) * topology.block_size, 0.f)
    , wet_output_(static_cast<size_t>(topology.output_channel_count) * topology.block_size, 0.f)
    , tone_output_(static_cast<size_t>(topology.output_channel_count) * topology.block_size, 0.f)
    , direct_output_(static_cast<size_t>(topology.output_channel_count) * topology.block_size, 0.f)
    , tc_filter_(nullptr)
    , transpose_(topology.transposed)
{
    input_gains_ = MakeDefaultInputRouting(input_channel_count_, order_);
    output_gains_ = MakeDefaultOutputRouting(order_, output_channel_count_);

    delay_bank_.SetDelays(std::vector<float>(order_, 500.f), block_size_);
}

FDN::FDN(uint32_t order, uint32_t block_size, bool transpose)
    : FDN(FDNTopology{.order = order, .block_size = block_size, .transposed = transpose})
{
}

FDN::FDN(FDN&& other) noexcept
    : delay_bank_(std::move(other.delay_bank_))
    , filter_bank_(std::move(other.filter_bank_))
    , mixing_matrix_(std::move(other.mixing_matrix_))
    , input_gains_(std::move(other.input_gains_))
    , output_gains_(std::move(other.output_gains_))
    , direct_path_(std::move(other.direct_path_))
    , order_(other.order_)
    , block_size_(other.block_size_)
    , input_channel_count_(other.input_channel_count_)
    , output_channel_count_(other.output_channel_count_)
    , direct_gain_(other.direct_gain_)
    , feedback_(std::move(other.feedback_))
    , temp_buffer_(std::move(other.temp_buffer_))
    , wet_output_(std::move(other.wet_output_))
    , tone_output_(std::move(other.tone_output_))
    , direct_output_(std::move(other.direct_output_))
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
        direct_path_ = std::move(other.direct_path_);
        feedback_ = std::move(other.feedback_);
        temp_buffer_ = std::move(other.temp_buffer_);
        wet_output_ = std::move(other.wet_output_);
        tone_output_ = std::move(other.tone_output_);
        direct_output_ = std::move(other.direct_output_);
        tc_filter_ = std::move(other.tc_filter_);
        order_ = other.order_;
        block_size_ = other.block_size_;
        input_channel_count_ = other.input_channel_count_;
        output_channel_count_ = other.output_channel_count_;
        direct_gain_ = other.direct_gain_;
        transpose_ = other.transpose_;
    }
    return *this;
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
    if (gains == nullptr || gains->InputChannelCount() != input_channel_count_ || gains->OutputChannelCount() != order_)
    {
        std::println(std::cerr, "Input routing must have {} input and {} output channels.", input_channel_count_,
                     order_);
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
    return SetInputGains(std::make_unique<ParallelGains>(ParallelGainsMode::Split, gains));
}

AudioProcessor* FDN::GetInputGains() const
{
    return input_gains_.get();
}

bool FDN::SetOutputGains(std::unique_ptr<AudioProcessor> gains)
{
    if (gains == nullptr || gains->InputChannelCount() != order_ ||
        gains->OutputChannelCount() != output_channel_count_)
    {
        std::println(std::cerr, "Output routing must have {} input and {} output channels.", order_,
                     output_channel_count_);
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
    return SetOutputGains(std::make_unique<ParallelGains>(ParallelGainsMode::Merge, gains));
}

AudioProcessor* FDN::GetOutputGains() const
{
    return output_gains_.get();
}

bool FDN::SetDirectPath(std::unique_ptr<AudioProcessor> direct)
{
    if (direct == nullptr)
    {
        direct_path_ = nullptr;
        return true;
    }

    if (direct->InputChannelCount() != input_channel_count_ || direct->OutputChannelCount() != output_channel_count_)
    {
        std::println(std::cerr, "Direct routing must have {} input and {} output channels.", input_channel_count_,
                     output_channel_count_);
        return false;
    }

    direct_path_ = std::move(direct);
    return true;
}

AudioProcessor* FDN::GetDirectPath() const
{
    return direct_path_.get();
}

void FDN::SetDirectGain(float gain)
{
    if (input_channel_count_ != output_channel_count_)
    {
        std::println(std::cerr, "Scalar direct gain requires matching input and output channel counts ({} and {}).",
                     input_channel_count_, output_channel_count_);
        return;
    }
    direct_path_ = nullptr;
    direct_gain_ = gain;
}

bool FDN::SetLoopFilter(std::unique_ptr<AudioProcessor> filter_bank)
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

AudioProcessor* FDN::GetLoopFilter() const
{
    return filter_bank_.get();
}

bool FDN::SetDelayBank(const DelayBankOptions& config)
{
    for (const auto& delay : config.delays)
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

    if (config.delays.size() != order_)
    {
        std::println(std::cerr, "Delays must have {} elements.", order_);
        return false;
    }

    delay_bank_ = DelayBank(config);
    return true;
}

bool FDN::SetDelays(const std::span<const float> delays, DelayInterpolationType interpolation_type)
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

    DelayBankOptions options;
    options.delays = std::vector<float>(delays.begin(), delays.end());
    options.block_size = block_size_;
    options.interpolation_type = interpolation_type;

    delay_bank_ = DelayBank(options);

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

    if (filter->InputChannelCount() != output_channel_count_ || filter->OutputChannelCount() != output_channel_count_)
    {
        std::println(std::cerr, "Tone correction must have {} input and output channels.", output_channel_count_);
        return false;
    }

    tc_filter_ = std::move(filter);
    return true;
}

AudioProcessor* FDN::GetTCFilter() const
{
    return tc_filter_.get();
}

void FDN::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == input_channel_count_);
    assert(output.ChannelCount() == output_channel_count_ ||
           (output_channel_count_ == 1U && output.ChannelCount() > 1U));
    assert(input_gains_ != nullptr);
    assert(output_gains_ != nullptr);

    const ScopedNoDenormals no_denormals;

    AudioBuffer routed_output = output_channel_count_ == 1U ? output.GetChannelBuffer(0) : output;
    if (transpose_)
    {
        TickTranspose(input, routed_output);
    }
    else
    {
        Tick(input, routed_output);
    }

    if (output_channel_count_ == 1U && output.ChannelCount() > 1U)
    {
        const auto mono_output = routed_output.GetChannelSpan(0);
        for (uint32_t channel = 1; channel < output.ChannelCount(); ++channel)
        {
            std::ranges::copy(mono_output, output.GetChannelSpan(channel).begin());
        }
    }
}

uint32_t FDN::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return input_channel_count_;
}

uint32_t FDN::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return output_channel_count_;
}

void FDN::PrepareOutput(const AudioBuffer& input, const AudioBuffer& wet_input) noexcept SFFDN_NONBLOCKING
{
    const uint32_t sample_count = input.SampleCount();
    const size_t output_sample_count = static_cast<size_t>(sample_count) * output_channel_count_;
    std::ranges::fill(std::span(wet_output_).first(output_sample_count), 0.f);
    AudioBuffer wet_output(sample_count, output_channel_count_, wet_output_);
    output_gains_->Process(wet_input, wet_output);

    if (direct_path_ != nullptr)
    {
        std::ranges::fill(std::span(direct_output_).first(output_sample_count), 0.f);
        AudioBuffer direct_output(sample_count, output_channel_count_, direct_output_);
        direct_path_->Process(input, direct_output);
    }
    else if (input_channel_count_ == output_channel_count_)
    {
        AudioBuffer direct_output(sample_count, output_channel_count_, direct_output_);
        for (uint32_t channel = 0; channel < output_channel_count_; ++channel)
        {
            ArrayMath::Scale(input.GetChannelSpan(channel), direct_gain_, direct_output.GetChannelSpan(channel));
        }
    }
    else
    {
        std::ranges::fill(std::span(direct_output_).first(output_sample_count), 0.f);
    }
}

void FDN::AccumulateOutput(AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    const uint32_t sample_count = output.SampleCount();
    const size_t output_sample_count = static_cast<size_t>(sample_count) * output_channel_count_;
    AudioBuffer wet_output(sample_count, output_channel_count_, wet_output_);
    AudioBuffer tone_output(sample_count, output_channel_count_, tone_output_);
    const AudioBuffer* processed_wet = &wet_output;
    if (tc_filter_ != nullptr)
    {
        std::ranges::fill(std::span(tone_output_).first(output_sample_count), 0.f);
        tc_filter_->Process(wet_output, tone_output);
        processed_wet = &tone_output;
    }

    AudioBuffer direct_output(sample_count, output_channel_count_, direct_output_);
    for (uint32_t channel = 0; channel < output_channel_count_; ++channel)
    {
        ArrayMath::Accumulate(output.GetChannelSpan(channel), processed_wet->GetChannelSpan(channel));
        ArrayMath::Accumulate(output.GetChannelSpan(channel), direct_output.GetChannelSpan(channel));
    }
}

void FDN::TickInternal(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() <= block_size_);

    const uint32_t block_size = input.SampleCount();
    const size_t internal_sample_count = static_cast<size_t>(block_size) * order_;

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

    PrepareOutput(input, feedback_buffer);

    mixing_matrix_->Process(feedback_buffer, temp_buffer);

    input_gains_->Process(input, feedback_buffer);
    ArrayMath::Add(std::span(feedback_).first(internal_sample_count),
                   std::span(temp_buffer_).first(internal_sample_count),
                   std::span(feedback_).first(internal_sample_count));

    delay_bank_.AddNextInputs(feedback_buffer);
    AccumulateOutput(output);
}

void FDN::TickTransposeInternal(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() <= block_size_);

    const uint32_t block_size = input.SampleCount();
    const size_t internal_sample_count = static_cast<size_t>(block_size) * order_;

    AudioBuffer temp_buffer(block_size, order_, temp_buffer_);
    AudioBuffer feedback_buffer(block_size, order_, feedback_);

    input_gains_->Process(input, temp_buffer);

    delay_bank_.GetNextOutputs(feedback_buffer);

    ArrayMath::Add(std::span(feedback_).first(internal_sample_count),
                   std::span(temp_buffer_).first(internal_sample_count),
                   std::span(feedback_).first(internal_sample_count));

    std::ranges::fill(std::span(temp_buffer_).first(internal_sample_count), 0.f);
    mixing_matrix_->Process(feedback_buffer, temp_buffer);

    if (filter_bank_)
    {
        std::ranges::fill(std::span(feedback_).first(internal_sample_count), 0.f);
        filter_bank_->Process(temp_buffer, feedback_buffer);
        std::swap(feedback_buffer, temp_buffer);
    }

    delay_bank_.AddNextInputs(temp_buffer);

    PrepareOutput(input, temp_buffer);
    AccumulateOutput(output);
}

void FDN::Tick(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    const uint32_t block_count = input.SampleCount() / block_size_;

    for (auto i = 0u; i < block_count; ++i)
    {
        const AudioBuffer input_block = input.Offset(i * block_size_, block_size_);
        AudioBuffer output_block = output.Offset(i * block_size_, block_size_);

        TickInternal(input_block, output_block);
    }

    const uint32_t remaining_samples = input.SampleCount() % block_size_;
    assert(block_size_ * block_count + remaining_samples == input.SampleCount());

    if (remaining_samples > 0)
    {
        const AudioBuffer input_block = input.Offset(block_count * block_size_, remaining_samples);
        AudioBuffer output_block = output.Offset(block_count * block_size_, remaining_samples);

        TickInternal(input_block, output_block);
    }
}

void FDN::TickTranspose(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    const uint32_t block_count = input.SampleCount() / block_size_;

    for (auto i = 0u; i < block_count; ++i)
    {
        const AudioBuffer input_block = input.Offset(i * block_size_, block_size_);
        AudioBuffer output_block = output.Offset(i * block_size_, block_size_);

        TickTransposeInternal(input_block, output_block);
    }

    const uint32_t remaining_samples = input.SampleCount() % block_size_;
    assert(block_size_ * block_count + remaining_samples == input.SampleCount());

    if (remaining_samples > 0)
    {
        const AudioBuffer input_block = input.Offset(block_count * block_size_, remaining_samples);
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
    if (direct_path_)
    {
        direct_path_->Clear();
    }
    if (tc_filter_)
    {
        tc_filter_->Clear();
    }

    std::ranges::fill(feedback_, 0.f);
    std::ranges::fill(temp_buffer_, 0.f);
    std::ranges::fill(wet_output_, 0.f);
    std::ranges::fill(tone_output_, 0.f);
    std::ranges::fill(direct_output_, 0.f);
}

std::unique_ptr<AudioProcessor> FDN::Clone() const
{
    return CloneFDN();
}

std::unique_ptr<FDN> FDN::CloneFDN() const
{
    auto clone = std::make_unique<FDN>(FDNTopology{
        .order = order_,
        .block_size = block_size_,
        .input_channel_count = input_channel_count_,
        .output_channel_count = output_channel_count_,
        .transposed = transpose_,
    });

    assert(input_gains_ != nullptr);
    assert(output_gains_ != nullptr);
    if (!clone->SetInputGains(input_gains_->Clone()) || !clone->SetOutputGains(output_gains_->Clone()))
    {
        throw std::logic_error("Failed to clone FDN boundary processors");
    }

    if (direct_path_ != nullptr)
    {
        if (!clone->SetDirectPath(direct_path_->Clone()))
        {
            throw std::logic_error("Failed to clone FDN direct path");
        }
    }
    else if (input_channel_count_ == output_channel_count_)
    {
        clone->SetDirectGain(direct_gain_);
    }

    if (!clone->SetLoopFilter(filter_bank_ ? filter_bank_->Clone() : nullptr))
    {
        throw std::logic_error("Failed to clone FDN loop filter");
    }
    clone->delay_bank_ = delay_bank_;
    if (!clone->SetFeedbackMatrix(mixing_matrix_ ? mixing_matrix_->Clone() : nullptr))
    {
        throw std::logic_error("Failed to clone FDN feedback matrix");
    }
    if (!clone->SetTCFilter(tc_filter_ ? tc_filter_->Clone() : nullptr))
    {
        throw std::logic_error("Failed to clone FDN tone correction");
    }

    clone->Clear();

    assert(clone->order_ == order_);
    assert(clone->block_size_ == block_size_);
    assert(clone->transpose_ == transpose_);
    assert(direct_path_ != nullptr || clone->direct_gain_ == direct_gain_);
    assert(clone->InputChannelCount() == InputChannelCount());
    assert(clone->OutputChannelCount() == OutputChannelCount());

    return clone;
}

} // namespace sfFDN