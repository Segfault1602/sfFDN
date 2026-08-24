#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>

#ifdef SFFDN_USE_IPP
#include "third_party/fea_ipp_process.h"
#endif

// The non-IPP implementation seems to be faster for now...
#ifdef SFFDN_USE_IPP
// #define SFFDN_USE_IPP_SPARSE_FIR
#endif

namespace sfFDN
{
#ifndef SFFDN_USE_IPP_SPARSE_FIR
class SparseFir::SparseFirImpl
{
  public:
    SparseFirImpl(const SparseFirOptions& config)
    {
        SetCoefficients(config);
    }

    void SetCoefficients(const SparseFirOptions& config)
    {
        coeffs_.clear();
        sparse_index_.clear();

        for (const auto& [index, coefficient] : config.coeffs)
        {
            coeffs_.push_back(coefficient);
            sparse_index_.push_back(index);
        }

        filter_order_ = *std::ranges::max_element(sparse_index_) + 1;

        delay_line_.SetMaximumDelay(filter_order_ + kDefaultBlockSize);
    }

    float Tick(float in) noexcept SFFDN_NONBLOCKING
    {
        delay_line_.Tick(in);

        float y = 0.f;
        for (size_t i = 0; i < coeffs_.size(); ++i)
        {
            const uint32_t tap = sparse_index_[i];
            y += coeffs_[i] * delay_line_.TapOut(tap);
        }

        return y;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == 1);

        delay_line_.AddNextInputs(input.GetChannelSpan(0));

        std::fill(output.GetChannelSpan(0).begin(), output.GetChannelSpan(0).end(), 0.f);
        delay_line_.GetNextOutputsAt(sparse_index_, output.GetChannelSpan(0), coeffs_);
    }

    constexpr uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return 1;
    }

    constexpr uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return 1;
    }

    void Clear()
    {
        delay_line_.Clear();
    }

    std::unique_ptr<SparseFirImpl> Clone() const
    {
        auto clone = std::make_unique<SparseFirImpl>(*this);
        // clone->coeffs_ = coeffs_;
        // clone->sparse_index_ = sparse_index_;
        // clone->filter_order_ = filter_order_;
        // clone->delay_line_ = delay_line_;

        // clone->Clear();
        return clone;
    }

    nlohmann::json ToJson() const
    {
        nlohmann::json j;
        j["type"] = "SparseFir";
        j["coefficients"] = coeffs_;
        j["indices"] = sparse_index_;
        return j;
    }

  private:
    std::vector<float> coeffs_;
    Delay delay_line_;

    std::vector<uint32_t> sparse_index_;
    uint32_t filter_order_{0};

    SparseFirImpl() = default;
};
#else
class SparseFir::SparseFirImpl
{
  public:
    SparseFirImpl() = default;

    ~SparseFirImpl()
    {
        Cleanup();
    }

    void Cleanup()
    {
        if (buffer_)
        {
            ippsFree(buffer_);
            buffer_ = nullptr;
        }
        state_ = nullptr;
    }

    void SetCoefficients(std::span<const float> coeffs, std::span<const uint32_t> indices)
    {
        assert(coeffs.size() == indices.size());
        coeffs_.assign(coeffs.begin(), coeffs.end());
        sparse_index_.assign(indices.begin(), indices.end());

        Cleanup();

        int tap_length = static_cast<int>(coeffs_.size());
        int buffer_size = 0;
        IppStatus status = ippsFIRSparseGetStateSize_32f(tap_length, sparse_index_.back(), &buffer_size);
        if (status != ippStsNoErr)
        {
            throw std::runtime_error("Failed to get FIR sparse state size");
        }
        buffer_ = ippsMalloc_8u(buffer_size);
        std::vector<Ipp32s> ipps_indices(sparse_index_.size());
        std::transform(sparse_index_.begin(), sparse_index_.end(), ipps_indices.begin(),
                       [](uint32_t idx) { return static_cast<Ipp32s>(idx); });
        status = ippsFIRSparseInit_32f(&state_, coeffs_.data(), ipps_indices.data(), tap_length, nullptr, buffer_);
        if (status != ippStsNoErr)
        {
            throw std::runtime_error("Failed to initialize FIR sparse state");
        }
    }

    float Tick(float in) noexcept SFFDN_NONBLOCKING
    {
        float out = 0.f;
        ippsFIRSparse_32f(&in, &out, 1, state_);
        return out;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == 1);

        ippsFIRSparse_32f(input.GetChannelSpan(0).data(), output.GetChannelSpan(0).data(),
                          static_cast<int>(input.SampleCount()), state_);
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return 1;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING
    {
        return 1;
    }

    void Clear()
    {
        std::vector<float> zero_delay(sparse_index_.back(), 0.f);
        ippsFIRSparseSetDlyLine_32f(state_, zero_delay.data());
    }

    std::unique_ptr<SparseFirImpl> Clone() const
    {
        auto clone = std::make_unique<SparseFirImpl>();
        clone->SetCoefficients(coeffs_, sparse_index_);
        return clone;
    }

  private:
    std::vector<float> coeffs_;
    std::vector<uint32_t> sparse_index_;

    IppsFIRSparseState_32f* state_{nullptr};
    Ipp8u* buffer_{nullptr};
};
#endif

SparseFir::SparseFir(const SparseFirOptions& config)
    : impl_(std::make_unique<SparseFirImpl>(config))
    , config_(config)
{
}

SparseFir::~SparseFir() = default;

void SparseFir::SetCoefficients(const SparseFirOptions& config)
{
    impl_->SetCoefficients(config);
}

float SparseFir::Tick(float in) noexcept SFFDN_NONBLOCKING
{
    return impl_->Tick(in);
}

void SparseFir::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    impl_->Process(input, output);
}

uint32_t SparseFir::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return impl_->InputChannelCount();
}

uint32_t SparseFir::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return impl_->OutputChannelCount();
}

void SparseFir::Clear()
{
    impl_->Clear();
}

std::unique_ptr<AudioProcessor> SparseFir::Clone() const
{
    auto clone = std::make_unique<SparseFir>(config_);
    return clone;
}

} // namespace sfFDN