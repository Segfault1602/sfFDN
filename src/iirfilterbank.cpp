#include "sffdn/filterbank.h"

#include "json_helper.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter.h"

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
#include <Accelerate/Accelerate.h>
#endif

#define IIRFILTERBANK_USE_EIGEN 0
#if IIRFILTERBANK_USE_EIGEN
#include <Eigen/Core>
#endif

namespace
{
#if IIRFILTERBANK_USE_EIGEN
class BiquadMC
{
  public:
    BiquadMC()
    {
        b0_ = Eigen::ArrayXf::Zero(1);
        b1_ = Eigen::ArrayXf::Zero(1);
        b2_ = Eigen::ArrayXf::Zero(1);
        a1_ = Eigen::ArrayXf::Zero(1);
        a2_ = Eigen::ArrayXf::Zero(1);

        state0_ = Eigen::ArrayXf::Zero(1);
        state1_ = Eigen::ArrayXf::Zero(1);
    }

    void SetCoefficients(uint32_t channel_count, std::span<const float> coeffs)
    {
        constexpr uint32_t coeff_per_stage = 5;
        assert(coeffs.size() == channel_count * 5);
        b0_ = Eigen::ArrayXf::Zero(channel_count);
        b1_ = Eigen::ArrayXf::Zero(channel_count);
        b2_ = Eigen::ArrayXf::Zero(channel_count);
        a1_ = Eigen::ArrayXf::Zero(channel_count);
        a2_ = Eigen::ArrayXf::Zero(channel_count);

        state0_ = Eigen::ArrayXf::Zero(channel_count);
        state1_ = Eigen::ArrayXf::Zero(channel_count);

        for (auto ch = 0u; ch < channel_count; ++ch)
        {
            auto coeffs_span = coeffs.subspan(ch * coeff_per_stage, coeff_per_stage);
            b0_(ch) = coeffs_span[0];
            b1_(ch) = coeffs_span[1];
            b2_(ch) = coeffs_span[2];
            a1_(ch) = coeffs_span[3];
            a2_(ch) = coeffs_span[4];
        }

        temp_ = Eigen::ArrayXf::Zero(channel_count);
    }

    template <typename Derived>
    void Process(const Eigen::ArrayBase<Derived>& x)
    {
        temp_ = b0_ * x + state0_;
        state0_ = b1_ * x + state1_ - a1_ * temp_;
        state1_ = b2_ * x - a2_ * temp_;

        const_cast<Eigen::ArrayBase<Derived>&>(x) = temp_;
    }

    void Clear()
    {
        state0_.setZero();
        state1_.setZero();
    }

  private:
    Eigen::ArrayXf b0_;
    Eigen::ArrayXf b1_;
    Eigen::ArrayXf b2_;
    Eigen::ArrayXf a1_;
    Eigen::ArrayXf a2_;

    Eigen::ArrayXf state0_;
    Eigen::ArrayXf state1_;

    Eigen::ArrayXf temp_;
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
        if (coeffs.size() % channel_count != 0)
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const uint32_t stage_count = static_cast<uint32_t>(coeffs.size() / channel_count);
#if IIRFILTERBANK_USE_EIGEN
        filters_.clear();
        channel_count_ = channel_count;
        temp_ = Eigen::ArrayXf::Zero(channel_count);
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
#else
        filters_.resize(channel_count);
        for (auto i = 0u; i < channel_count; ++i)
        {
            auto coeffs_span = coeffs.subspan(i * stage_count, stage_count);
            filters_[i].SetCoefficients(coeffs_span);
        }
#endif
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());

#if !IIRFILTERBANK_USE_EIGEN
        assert(input.ChannelCount() == filters_.size());
        for (auto i = 0u; i < filters_.size(); ++i)
        {
            auto input_buf = input.GetChannelBuffer(i);
            auto output_buf = output.GetChannelBuffer(i);
            filters_[i].Process(input_buf, output_buf);
        }
#else
        assert(input.ChannelCount() == channel_count_);
        Eigen::Map<const Eigen::ArrayXXf> in(input.Data(), input.SampleCount(), input.ChannelCount());
        Eigen::Map<Eigen::ArrayXXf> out(output.Data(), output.SampleCount(), output.ChannelCount());

        for (auto i = 0u; i < input.SampleCount(); ++i)
        {
            temp_ = in.row(i);

            for (auto j = 0u; j < filters_.size(); ++j)
            {
                filters_[j].Process(temp_);
            }

            out.row(i) = temp_;
        }
#endif
    }

    uint32_t InputChannelCount() const
    {
#if IIRFILTERBANK_USE_EIGEN
        return channel_count_;
#else
        return filters_.size();
#endif
    }

    uint32_t OutputChannelCount() const
    {
#if IIRFILTERBANK_USE_EIGEN
        return channel_count_;
#else
        return filters_.size();
#endif
    }

  private:
#if IIRFILTERBANK_USE_EIGEN
    std::vector<BiquadMC> filters_;
    uint32_t channel_count_;
    Eigen::ArrayXf temp_;
#else
    std::vector<CascadedBiquads> filters_;
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

    void SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
    {
        if (coeffs.size() % channel_count != 0)
        {
            throw std::runtime_error("Invalid coefficient size");
        }

        const uint32_t stage_count = static_cast<uint32_t>(coeffs.size() / channel_count);

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

void IIRFilterBank::SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
{
    impl_->SetFilter(coeffs, channel_count);
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

nlohmann::json IIRFilterBank::ToJson() const
{
    nlohmann::json j;
    j["type"] = "IIRFilterBank";
    j["channel_count_"] = channel_count_;
    j["coeffs_"] = nlohmann::json::array();
    for (const auto& coeffs : coeffs_)
    {
        std::array<float, 6> coeffs_array{coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a0, coeffs.a1, coeffs.a2};
        j["coeffs_"].push_back(coeffs_array);
    }
    return j;
}

std::unique_ptr<IIRFilterBank> IIRFilterBank::FromJson(const nlohmann::json& j)
{
    ThrowIfNotType(j, "IIRFilterBank");

    auto channel_count = j["channel_count_"].get<uint32_t>();

    const auto& coeffs_json = j["coeffs_"];
    if (!coeffs_json.is_array())
    {
        throw std::invalid_argument("JSON 'coeffs_' field must be an array.");
    }

    std::vector<FilterCoefficients> coeffs;
    for (const auto& coeffs_entry : coeffs_json)
    {
        if (!coeffs_entry.is_array() || coeffs_entry.size() != 6)
        {
            throw std::invalid_argument("Each entry in 'coeffs_' must be an array of 6 floats.");
        }

        FilterCoefficients filter_coeffs;
        filter_coeffs.b0 = coeffs_entry[0].get<float>();
        filter_coeffs.b1 = coeffs_entry[1].get<float>();
        filter_coeffs.b2 = coeffs_entry[2].get<float>();
        filter_coeffs.a0 = coeffs_entry[3].get<float>();
        filter_coeffs.a1 = coeffs_entry[4].get<float>();
        filter_coeffs.a2 = coeffs_entry[5].get<float>();

        coeffs.push_back(filter_coeffs);
    }

    auto filter_bank = std::make_unique<IIRFilterBank>();
    filter_bank->SetFilter(coeffs, channel_count);

    return filter_bank;
}

} // namespace sfFDN