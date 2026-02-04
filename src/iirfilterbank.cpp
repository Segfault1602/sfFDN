#include "sffdn/filterbank.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

// #define IIRFILTERBANK_USE_EIGEN 1
#if IIRFILTERBANK_USE_EIGEN
#include <Eigen/Core>
#endif

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
#if IIRFILTERBANK_USE_EIGEN
        state1_.setZero();
        state2_.setZero();
#endif
    }

    void SetFilter(std::span<float> coeffs, uint32_t channel_count, uint32_t stage_count)
    {
        uint32_t coeff_per_stage = 0;
        if (coeffs.size() == channel_count * stage_count * 5)
        {
            coeff_per_stage = 5;
        }
        else if (coeffs.size() == channel_count * stage_count * 6)
        {
            coeff_per_stage = 6;
        }
        else
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const uint32_t coeffs_per_channel = coeff_per_stage * stage_count;

#if IIRFILTERBANK_USE_EIGEN
        b0_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
        b1_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
        b2_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
        a1_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
        a2_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);

        state1_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
        state2_ = Eigen::ArrayXXf::Zero(channel_count, stage_count);
#endif

        filters_.resize(channel_count);
        for (auto i = 0u; i < channel_count; ++i)
        {
            auto coeffs_span = coeffs.subspan(i * coeffs_per_channel, coeffs_per_channel);
            filters_[i].SetCoefficients(stage_count, coeffs_span);
#if IIRFILTERBANK_USE_EIGEN
            for (auto j = 0u; j < stage_count; ++j)
            {
                auto stage_coeffs = coeffs_span.subspan(j * coeff_per_stage, coeff_per_stage);
                if (coeff_per_stage == 6)
                {
                    b0_(i, j) = stage_coeffs[0] / stage_coeffs[3];
                    b1_(i, j) = stage_coeffs[1] / stage_coeffs[3];
                    b2_(i, j) = stage_coeffs[2] / stage_coeffs[3];
                    a1_(i, j) = stage_coeffs[4] / stage_coeffs[3];
                    a2_(i, j) = stage_coeffs[5] / stage_coeffs[3];
                }
                else
                {
                    b0_(i, j) = stage_coeffs[0];
                    b1_(i, j) = stage_coeffs[1];
                    b2_(i, j) = stage_coeffs[2];
                    a1_(i, j) = stage_coeffs[3];
                    a2_(i, j) = stage_coeffs[4];
                }
            }
#endif
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == filters_.size());

#if !IIRFILTERBANK_USE_EIGEN
        for (auto i = 0u; i < filters_.size(); ++i)
        {
            auto input_buf = input.GetChannelBuffer(i);
            auto output_buf = output.GetChannelBuffer(i);
            filters_[i].Process(input_buf, output_buf);
        }
#else
        Eigen::Array<float, Eigen::Dynamic, 1, Eigen::AutoAlign> in(input.ChannelCount());
        Eigen::Array<float, Eigen::Dynamic, 1, Eigen::AutoAlign> out(input.ChannelCount());
        for (auto i = 0u; i < input.SampleCount(); ++i)
        {
            for (auto n = 0u; n < input.ChannelCount(); ++n)
            {
                in(n) = input.GetChannelSpan(n)[i];
            }
            for (auto stage = 0u; stage < b0_.cols(); ++stage)
            {
                out = b0_.col(stage) * in + state1_.col(stage);
                state1_.col(stage) = b1_.col(stage) * in + state2_.col(stage) - a1_.col(stage) * out;
                state2_.col(stage) = b2_.col(stage) * in - a2_.col(stage) * out;
                in = out;
            }
            for (auto n = 0u; n < input.ChannelCount(); ++n)
            {
                output.GetChannelSpan(n)[i] = out(n);
            }
        }
#endif
    }

    uint32_t InputChannelCount() const
    {
        return filters_.size();
    }

    uint32_t OutputChannelCount() const
    {
        return filters_.size();
    }

  private:
    std::vector<CascadedBiquads> filters_;
#if IIRFILTERBANK_USE_EIGEN
    Eigen::ArrayXXf b0_;
    Eigen::ArrayXXf b1_;
    Eigen::ArrayXXf b2_;
    Eigen::ArrayXXf a1_;
    Eigen::ArrayXXf a2_;

    Eigen::ArrayXXf state1_;
    Eigen::ArrayXXf state2_;
#endif
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

        uint32_t stage_count = coeffs_d_.size() / (channel_count_ * 5);
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
        vDSP_biquadm_ResetState(biquad_setup_);
    }

    void SetFilter(std::span<float> coeffs, uint32_t channel_count, uint32_t stage_count)
    {
        uint32_t coeff_per_stage = 0;
        if (coeffs.size() == channel_count * stage_count * 5)
        {
            coeff_per_stage = 5;
        }
        else if (coeffs.size() == channel_count * stage_count * 6)
        {
            coeff_per_stage = 6;
        }
        else
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const uint32_t coeffs_per_channel = coeff_per_stage * stage_count;

        coeffs_d_.reserve(coeffs.size());
        for (auto j = 0u; j < stage_count; ++j)
        {
            for (auto i = 0u; i < channel_count; ++i)
            {
                auto coeffs_span = coeffs.subspan((i * coeffs_per_channel) + (j * coeff_per_stage), coeff_per_stage);
                if (coeff_per_stage == 6)
                {
                    // vDSP_biquadm expects 5 coefficient per stage
                    coeffs_d_.push_back(static_cast<double>(coeffs_span[0]) / static_cast<double>(coeffs_span[3]));
                    coeffs_d_.push_back(static_cast<double>(coeffs_span[1]) / static_cast<double>(coeffs_span[3]));
                    coeffs_d_.push_back(static_cast<double>(coeffs_span[2]) / static_cast<double>(coeffs_span[3]));
                    coeffs_d_.push_back(static_cast<double>(coeffs_span[4]) / static_cast<double>(coeffs_span[3]));
                    coeffs_d_.push_back(static_cast<double>(coeffs_span[5]) / static_cast<double>(coeffs_span[3]));
                }
                else
                {
                    for (float j : coeffs_span)
                    {
                        coeffs_d_.push_back(static_cast<double>(j));
                    }
                }
            }
        }

        assert(coeffs_d_.size() == channel_count * stage_count * 5);

        biquad_setup_ = vDSP_biquadm_CreateSetup(coeffs_d_.data(), stage_count, channel_count);
        if (biquad_setup_ == nullptr)
        {
            throw std::runtime_error("Failed to create vDSP biquad setup");
        }

        vDSP_biquadm_SetCoefficientsDouble(biquad_setup_, coeffs_d_.data(), 0, 0, stage_count, channel_count_);

        channel_count_ = channel_count;
        input_ptrs_.resize(channel_count_);
        output_ptrs_.resize(channel_count_);
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept
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

    uint32_t InputChannelCount() const
    {
        return channel_count_;
    }

    uint32_t OutputChannelCount() const
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
{
}

IIRFilterBank& IIRFilterBank::operator=(IIRFilterBank&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

IIRFilterBank::~IIRFilterBank() = default;

void IIRFilterBank::Clear()
{
    impl_->Clear();
}

void IIRFilterBank::SetFilter(std::span<float> coeffs, uint32_t channel_count, size_t stage_count)
{
    impl_->SetFilter(coeffs, channel_count, stage_count);
}

void IIRFilterBank::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    impl_->Process(input, output);
}

uint32_t IIRFilterBank::InputChannelCount() const
{
    return impl_->InputChannelCount();
}

uint32_t IIRFilterBank::OutputChannelCount() const
{
    return impl_->OutputChannelCount();
}

std::unique_ptr<AudioProcessor> IIRFilterBank::Clone() const
{
    auto clone = std::make_unique<IIRFilterBank>();

    clone->impl_ = std::make_unique<IIRFilterBank::IIRFilterBankImpl>(*impl_);
    clone->impl_->Clear();

    return clone;
}

} // namespace sfFDN