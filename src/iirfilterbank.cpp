#include "sffdn/filterbank.h"

#include "json_helper.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter.h"
#include "simd_biquad_bank.h"

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

namespace sfFDN
{

// A single channel-parallel SIMD implementation is used on every platform. It was measured to
// beat both the per-channel vDSP_biquad path and the multichannel vDSP_biquadm path on Apple
// Silicon at every benchmarked order and cascade length, so no vendor-specific backend remains.
class IIRFilterBank::IIRFilterBankImpl
{
  public:
    IIRFilterBankImpl() = default;

    void Clear()
    {
        bank_.Clear();
    }

    void SetFilter(std::span<const FilterCoefficients> coeffs, uint32_t channel_count)
    {
        bank_.SetCoefficients(coeffs, channel_count);
        channel_count_ = channel_count;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == channel_count_);

        if (bank_.ChannelCount() != 0)
        {
            bank_.Process(input, output);
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
    uint32_t channel_count_{0};
    SimdBiquadBank bank_;
};

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