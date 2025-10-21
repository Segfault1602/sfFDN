#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <Eigen/Core>

#include <cassert>
#include <numeric>

namespace sfFDN
{
void Fir::SetCoefficients(std::span<const float> coeffs)
{
    coeffs_.assign(coeffs.begin(), coeffs.end());
    delay_line_.resize(coeffs_.size() * 2, 0.f);
    delay_index_ = 0;
}

float Fir::Tick(float in)
{
    delay_line_[delay_index_] = in;
    delay_line_[delay_index_ + coeffs_.size()] = in;

    auto delay_span = std::span(delay_line_).subspan(delay_index_, coeffs_.size());

    Eigen::Map<const Eigen::VectorXf> coeffs_map(coeffs_.data(), static_cast<Eigen::Index>(coeffs_.size()));
    Eigen::Map<const Eigen::VectorXf> delay_map(delay_span.data(), static_cast<Eigen::Index>(delay_span.size()));

    float y = coeffs_map.dot(delay_map);

    delay_index_ = (delay_index_ == 0) ? coeffs_.size() - 1 : delay_index_ - 1;
    return y;
}

void Fir::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    const uint32_t sample_count = input.SampleCount();
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1);

    for (uint32_t n = 0; n < sample_count; ++n)
    {
        output.GetChannelSpan(0)[n] = Tick(input.GetChannelSpan(0)[n]);
    }
}

uint32_t Fir::InputChannelCount() const
{
    return 1;
}

uint32_t Fir::OutputChannelCount() const
{
    return 1;
}

void Fir::Clear()
{
    std::fill(delay_line_.begin(), delay_line_.end(), 0.f);
    delay_index_ = 0;
}

std::unique_ptr<AudioProcessor> Fir::Clone() const
{
    auto clone = std::make_unique<Fir>();
    clone->SetCoefficients(coeffs_);
    return clone;
}

} // namespace sfFDN