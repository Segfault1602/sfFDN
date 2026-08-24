#include "sffdn/filterbank.h"

#include "json_helper.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef SFFDN_USE_VDSP
#include "third_party/fea_vdsp_process.h"
#endif

namespace
{
#ifndef SFFDN_USE_VDSP
class BiquadMC
{
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
    static void ProcessChannels(float* __restrict samples, const float* __restrict b0, const float* __restrict b1,
                                const float* __restrict b2, const float* __restrict a1, const float* __restrict a2,
                                float* __restrict state0, float* __restrict state1,
                                size_t channel_count) noexcept SFFDN_NONBLOCKING
    {
        for (auto ch = 0u; ch < channel_count; ++ch)
        {
            const float input = samples[ch];
            const float output = (b0[ch] * input) + state0[ch];
            state0[ch] = (b1[ch] * input) + state1[ch] - (a1[ch] * output);
            state1[ch] = (b2[ch] * input) - (a2[ch] * output);
            samples[ch] = output;
        }
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  public:
    void SetCoefficients(uint32_t channel_count, std::span<const float> coeffs)
    {
        constexpr uint32_t kCoeffPerStage = 5;
        assert(coeffs.size() == channel_count * kCoeffPerStage);
        b0_.assign(channel_count, 0.f);
        b1_.assign(channel_count, 0.f);
        b2_.assign(channel_count, 0.f);
        a1_.assign(channel_count, 0.f);
        a2_.assign(channel_count, 0.f);

        state0_.assign(channel_count, 0.f);
        state1_.assign(channel_count, 0.f);

        for (auto ch = 0u; ch < channel_count; ++ch)
        {
            auto coeffs_span = coeffs.subspan(ch * kCoeffPerStage, kCoeffPerStage);
            b0_[ch] = coeffs_span[0];
            b1_[ch] = coeffs_span[1];
            b2_[ch] = coeffs_span[2];
            a1_[ch] = coeffs_span[3];
            a2_[ch] = coeffs_span[4];
        }
    }

    void Process(std::span<float> x) noexcept SFFDN_NONBLOCKING
    {
        assert(x.size() == b0_.size());
        ProcessChannels(x.data(), b0_.data(), b1_.data(), b2_.data(), a1_.data(), a2_.data(), state0_.data(),
                        state1_.data(), x.size());
    }

    void Clear()
    {
        std::ranges::fill(state0_, 0.f);
        std::ranges::fill(state1_, 0.f);
    }

  private:
    std::vector<float> b0_;
    std::vector<float> b1_;
    std::vector<float> b2_;
    std::vector<float> a1_;
    std::vector<float> a2_;

    std::vector<float> state0_;
    std::vector<float> state1_;
};
#endif
} // namespace

namespace sfFDN
{

#ifndef SFFDN_USE_VDSP
class IIRFilterBank::IIRFilterBankImpl
{
  public:
    IIRFilterBankImpl() = default;

    void Clear()
    {
        for (auto& filter : filters_)
        {
            filter.Clear();
        }
    }

    void SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
    {
        if (channel_count == 0 || coeffs.size() % channel_count != 0)
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const auto stage_count = static_cast<uint32_t>(coeffs.size() / channel_count);
        filters_.clear();
        filters_.reserve(stage_count);
        channel_count_ = channel_count;
        temp_.assign(channel_count, 0.f);
        input_channels_.resize(channel_count);
        output_channels_.resize(channel_count);
        for (auto j = 0u; j < stage_count; ++j)
        {
            std::vector<float> biquads_coeffs(5 * channel_count);
            for (auto ch = 0u; ch < channel_count; ++ch)
            {
                // auto coeffs_span = coeffs.subspan((ch * coeffs_per_channel) + (j * coeff_per_stage),
                // coeff_per_stage);
                auto norm_coeffs = coeffs[(ch * stage_count) + j].Normalize();
                biquads_coeffs[ch * 5 + 0] = norm_coeffs.b0;
                biquads_coeffs[ch * 5 + 1] = norm_coeffs.b1;
                biquads_coeffs[ch * 5 + 2] = norm_coeffs.b2;
                biquads_coeffs[ch * 5 + 3] = norm_coeffs.a1;
                biquads_coeffs[ch * 5 + 4] = norm_coeffs.a2;
            }
            filters_.emplace_back();
            filters_.back().SetCoefficients(channel_count, biquads_coeffs);
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == channel_count_);
        assert(temp_.size() == channel_count_);
        assert(input_channels_.size() == channel_count_);
        assert(output_channels_.size() == channel_count_);
        for (auto ch = 0u; ch < channel_count_; ++ch)
        {
            input_channels_[ch] = input.GetChannelSpan(ch);
            output_channels_[ch] = output.GetChannelSpan(ch);
        }

        for (auto sample = 0u; sample < input.SampleCount(); ++sample)
        {
            for (auto ch = 0u; ch < channel_count_; ++ch)
            {
                temp_[ch] = input_channels_[ch][sample];
            }
            for (auto& filter : filters_)
            {
                filter.Process(temp_);
            }
            for (auto ch = 0u; ch < channel_count_; ++ch)
            {
                output_channels_[ch][sample] = temp_[ch];
            }
        }
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return channel_count_;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return channel_count_;
    }

  private:
    std::vector<BiquadMC> filters_;
    uint32_t channel_count_{0};
    std::vector<float> temp_;
    std::vector<std::span<const float>> input_channels_;
    std::vector<std::span<float>> output_channels_;
};
#else
class IIRFilterBank::IIRFilterBankImpl
{
  public:
    IIRFilterBankImpl()
        : channel_count_(0)
        , biquad_setup_(nullptr)
    {
    }

    ~IIRFilterBankImpl()
    {
        if (biquad_setup_ != nullptr)
        {
            vDSP_biquadm_DestroySetup(biquad_setup_);
            biquad_setup_ = nullptr;
        }
    }

    IIRFilterBankImpl(const IIRFilterBankImpl& other)
        : channel_count_(other.channel_count_)
        , coeffs_d_(other.coeffs_d_)
        , input_ptrs_(other.input_ptrs_)
        , output_ptrs_(other.output_ptrs_)
    {
        if (coeffs_d_.empty())
        {
            biquad_setup_ = nullptr;
            return;
        }

        const uint32_t stage_count = coeffs_d_.size() / (channel_count_ * 5);
        biquad_setup_ = vDSP_biquadm_CreateSetup(coeffs_d_.data(), stage_count, channel_count_);
    }

    IIRFilterBankImpl& operator=(const IIRFilterBankImpl& other)
    {
        if (this != &other)
        {
            *this = IIRFilterBankImpl(other);
        }
        return *this;
    }

    IIRFilterBankImpl(IIRFilterBankImpl&& other) noexcept
        : channel_count_(other.channel_count_)
        , biquad_setup_(other.biquad_setup_)
        , coeffs_d_(std::move(other.coeffs_d_))
        , input_ptrs_(std::move(other.input_ptrs_))
        , output_ptrs_(std::move(other.output_ptrs_))
    {
        other.biquad_setup_ = nullptr;
    }

    IIRFilterBankImpl& operator=(IIRFilterBankImpl&& other) noexcept
    {
        if (this != &other)
        {
            channel_count_ = other.channel_count_;
            biquad_setup_ = other.biquad_setup_;
            coeffs_d_ = std::move(other.coeffs_d_);
            input_ptrs_ = std::move(other.input_ptrs_);
            output_ptrs_ = std::move(other.output_ptrs_);
            other.biquad_setup_ = nullptr;
        }
        return *this;
    }

    void Clear()
    {
        if (biquad_setup_ != nullptr)
        {
            vDSP_biquadm_ResetState(biquad_setup_);
        }
    }

    void SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
    {
        if (channel_count == 0 || coeffs.size() % channel_count != 0)
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const auto stage_count = static_cast<uint32_t>(coeffs.size() / channel_count);

        if (biquad_setup_ != nullptr)
        {
            vDSP_biquadm_DestroySetup(biquad_setup_);
            biquad_setup_ = nullptr;
        }

        channel_count_ = channel_count;
        coeffs_d_.clear();
        coeffs_d_.reserve(coeffs.size());
        for (auto j = 0u; j < stage_count; ++j)
        {
            for (auto i = 0u; i < channel_count; ++i)
            {
                auto norm_coeffs = coeffs[(j + i * stage_count)].Normalize();
                // auto coeffs_span = coeffs.subspan((i * coeffs_per_channel) + (j * coeff_per_stage), coeff_per_stage);

                coeffs_d_.push_back(static_cast<double>(norm_coeffs.b0));
                coeffs_d_.push_back(static_cast<double>(norm_coeffs.b1));
                coeffs_d_.push_back(static_cast<double>(norm_coeffs.b2));
                coeffs_d_.push_back(static_cast<double>(norm_coeffs.a1));
                coeffs_d_.push_back(static_cast<double>(norm_coeffs.a2));
            }
        }

        assert(coeffs_d_.size() == channel_count * stage_count * 5);

        biquad_setup_ = vDSP_biquadm_CreateSetup(coeffs_d_.data(), stage_count, channel_count);
        if (biquad_setup_ == nullptr)
        {
            throw std::runtime_error("Failed to create vDSP biquad setup");
        }

        vDSP_biquadm_SetCoefficientsDouble(biquad_setup_, coeffs_d_.data(), 0, 0, stage_count, channel_count_);

        input_ptrs_.resize(channel_count_);
        output_ptrs_.resize(channel_count_);
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == channel_count_);
        assert(biquad_setup_ != nullptr);
        assert(input_ptrs_.size() == channel_count_);
        assert(output_ptrs_.size() == channel_count_);

        for (auto i = 0u; i < channel_count_; ++i)
        {
            input_ptrs_[i] = input.GetChannelSpan(i).data();
            output_ptrs_[i] = output.GetChannelSpan(i).data();
        }

        vDSP_biquadm(biquad_setup_, input_ptrs_.data(), 1, output_ptrs_.data(), 1, input.SampleCount());
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return channel_count_;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return channel_count_;
    }

  private:
    uint32_t channel_count_;
    vDSP_biquadm_Setup biquad_setup_;

    std::vector<double> coeffs_d_;

    std::vector<const float*> input_ptrs_;
    std::vector<float*> output_ptrs_;
};

#endif

IIRFilterBank::IIRFilterBank()
    : impl_(std::make_unique<IIRFilterBankImpl>())
{
}

IIRFilterBank::IIRFilterBank(IIRFilterBank&& other) noexcept
    : impl_(std::move(other.impl_))
    , coeffs_(std::move(other.coeffs_))
{
}

IIRFilterBank& IIRFilterBank::operator=(IIRFilterBank&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
        coeffs_ = std::move(other.coeffs_);
    }
    return *this;
}

IIRFilterBank::~IIRFilterBank() = default;

void IIRFilterBank::Clear()
{
    impl_->Clear();
}

void IIRFilterBank::SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
{
    impl_->SetFilter(coeffs, channel_count);
    coeffs_.assign(coeffs.begin(), coeffs.end());
}

void IIRFilterBank::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    impl_->Process(input, output);
}

uint32_t IIRFilterBank::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return impl_->InputChannelCount();
}

uint32_t IIRFilterBank::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return impl_->OutputChannelCount();
}

std::unique_ptr<AudioProcessor> IIRFilterBank::Clone() const
{
    auto clone = std::make_unique<IIRFilterBank>();
    if (InputChannelCount() != 0)
    {
        clone->SetFilter(coeffs_, InputChannelCount());
    }
    return clone;
}

} // namespace sfFDN