#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#ifdef SFFDN_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

#ifdef SFFDN_USE_IPP
#include <ipp.h>
#endif

namespace sfFDN
{
#ifndef SFFDN_USE_IPP
class Fir::FirImpl
{
  public:
    FirImpl() = default;

    void SetCoefficients(std::span<const float> coeffs)
    {
        coeffs_.assign(coeffs.begin(), coeffs.end());
        delay_line_.resize(coeffs_.size() * 2, 0.f);
        delay_index_ = 0;
    }

    float Tick(float in)
    {
        delay_line_[delay_index_] = in;
        delay_line_[delay_index_ + coeffs_.size()] = in;

        auto delay_span = std::span(delay_line_).subspan(delay_index_, coeffs_.size());

        const Eigen::Map<const Eigen::VectorXf> coeffs_map(coeffs_.data(), static_cast<Eigen::Index>(coeffs_.size()));
        const Eigen::Map<const Eigen::VectorXf> delay_map(delay_span.data(),
                                                          static_cast<Eigen::Index>(delay_span.size()));

        const float y = coeffs_map.dot(delay_map);

        delay_index_ = (delay_index_ == 0) ? coeffs_.size() - 1 : delay_index_ - 1;
        return y;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept
    {
        const uint32_t sample_count = input.SampleCount();
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == 1);

        for (uint32_t n = 0; n < sample_count; ++n)
        {
            output.GetChannelSpan(0)[n] = Tick(input.GetChannelSpan(0)[n]);
        }
    }

    constexpr uint32_t InputChannelCount() const
    {
        return 1;
    }

    constexpr uint32_t OutputChannelCount() const
    {
        return 1;
    }

    void Clear()
    {
        std::ranges::fill(delay_line_, 0.f);
        delay_index_ = 0;
    }

    std::unique_ptr<FirImpl> Clone() const
    {
        auto clone = std::make_unique<FirImpl>();
        clone->SetCoefficients(coeffs_);
        return clone;
    }

  private:
    std::vector<float> coeffs_;
    std::vector<float> delay_line_;
    uint32_t delay_index_{0};
};
#else
class Fir::FirImpl
{
  public:
    FirImpl() = default;

    ~FirImpl()
    {
        Cleanup();
    }

    void Cleanup()
    {
        if (spec_ != nullptr)
        {
            ippsFree(spec_);
            spec_ = nullptr;
        }
        if (buffer_ != nullptr)
        {
            ippsFree(buffer_);
            buffer_ = nullptr;
        }
        if (taps_ != nullptr)
        {
            ippsFree(taps_);
            taps_ = nullptr;
        }
        if (delay_line_ != nullptr)
        {
            ippsFree(delay_line_);
            delay_line_ = nullptr;
        }
        if (source_delay_ != nullptr)
        {
            ippsFree(source_delay_);
            source_delay_ = nullptr;
        }
    }

    void SetCoefficients(std::span<const float> coeffs)
    {
        if (coeffs.empty())
        {
            throw std::invalid_argument("FirImpl: Tap length must be greater than zero");
        }

        const int tap_length = static_cast<int>(coeffs.size());

        int spec_size = 0;
        int buffer_size = 0;
        IppStatus status = ippsFIRSRGetSize(tap_length, ipp32f, &spec_size, &buffer_size);

        if (status != ippStsNoErr)
        {
            throw std::runtime_error("FirImpl: Failed to get FIR spec size");
        }

        Cleanup();
        tap_length_ = tap_length;
        spec_ = reinterpret_cast<IppsFIRSpec_32f*>(ippsMalloc_8u(spec_size));
        buffer_ = ippsMalloc_8u(buffer_size);
        taps_ = ippsMalloc_32f(tap_length);
        delay_line_ = ippsMalloc_32f(tap_length - 1);
        source_delay_ = ippsMalloc_32f(tap_length - 1);

        ippsCopy_32f(coeffs.data(), taps_, tap_length);
        ippsZero_32f(delay_line_, tap_length - 1);
        ippsZero_32f(source_delay_, tap_length - 1);

        status = ippsFIRSRInit_32f(taps_, tap_length, ippAlgAuto, spec_);
        if (status != ippStsNoErr)
        {
            throw std::runtime_error("FirImpl: Failed to initialize FIR spec");
        }
    }

    float Tick(float in)
    {
        float out = 0.f;
        ippsFIRSR_32f(&in, &out, 1, spec_, source_delay_, delay_line_, buffer_);
        std::swap(source_delay_, delay_line_);
        return out;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept
    {
        const uint32_t sample_count = input.SampleCount();
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == 1);

        ippsFIRSR_32f(input.GetChannelSpan(0).data(), output.GetChannelSpan(0).data(), static_cast<int>(sample_count),
                      spec_, source_delay_, delay_line_, buffer_);
        std::swap(source_delay_, delay_line_);
    }

    uint32_t InputChannelCount() const
    {
        return 1;
    }

    uint32_t OutputChannelCount() const
    {
        return 1;
    }

    void Clear()
    {
        ippsZero_32f(delay_line_, tap_length_ - 1);
    }

    std::unique_ptr<FirImpl> Clone() const
    {
        auto clone = std::make_unique<FirImpl>();

        if (tap_length_ > 0 && (taps_ != nullptr))
        {
#pragma clang unsafe_buffer_usage begin
            clone->SetCoefficients(std::span<const float>(taps_, tap_length_));
#pragma clang unsafe_buffer_usage end
        }

        return clone;
    }

  private:
    int tap_length_{0};
    float* taps_{nullptr};
    float* delay_line_{nullptr};
    float* source_delay_{nullptr};
    Ipp8u* buffer_{nullptr};
    IppsFIRSpec_32f* spec_{nullptr};
};
#endif

Fir::Fir(const FirOptions& config)
    : impl_(std::make_unique<FirImpl>())
{
    SetCoefficients(config.coeffs);
}

Fir::~Fir() = default;

Fir::Fir(const Fir& other)
    : impl_(other.impl_->Clone())
{
}

Fir& Fir::operator=(const Fir& other)
{
    if (this != &other)
    {
        impl_ = other.impl_->Clone();
    }
    return *this;
}

Fir::Fir(Fir&& other) noexcept
    : impl_(std::move(other.impl_))
{
}

Fir& Fir::operator=(Fir&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void Fir::SetCoefficients(std::span<const float> coeffs)
{
    impl_->SetCoefficients(coeffs);
}

float Fir::Tick(float in)
{
    return impl_->Tick(in);
}

void Fir::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    impl_->Process(input, output);
}

uint32_t Fir::InputChannelCount() const
{
    return impl_->InputChannelCount();
}

uint32_t Fir::OutputChannelCount() const
{
    return impl_->OutputChannelCount();
}

void Fir::Clear()
{
    impl_->Clear();
}

std::unique_ptr<AudioProcessor> Fir::Clone() const
{
    auto clone = std::make_unique<Fir>();
    clone->impl_ = impl_->Clone();
    return clone;
}

std::unique_ptr<AudioProcessor> MakeFirFilter(const FirOptions& config, float sparse_threshold)
{
    if (config.coeffs.empty())
    {
        throw std::invalid_argument("MakeFirFilter: Coefficients cannot be empty");
    }

    const size_t non_zero_count =
        std::count_if(config.coeffs.begin(), config.coeffs.end(), [](float coeff) { return std::abs(coeff) > 1e-6f; });
    const float sparsity = static_cast<float>(non_zero_count) / static_cast<float>(config.coeffs.size());

    if (sparsity <= sparse_threshold)
    {
        SparseFirOptions sparse_config;
        for (size_t i = 0; i < config.coeffs.size(); ++i)
        {
            if (std::abs(config.coeffs[i]) > 1e-6f)
            {
                sparse_config.coeffs.emplace_back(i, config.coeffs[i]);
            }
        }
        return std::make_unique<SparseFir>(sparse_config);
    }

    return std::make_unique<Fir>(config);
}
} // namespace sfFDN