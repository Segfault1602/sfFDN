#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay.h"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace
{
constexpr uint32_t kDefaultBlockSize = 512; // Arbitrary block size, should be configurable at some point
}

namespace sfFDN
{

void SparseFir::SetCoefficients(std::span<const float> coeffs, std::span<const uint32_t> indices)
{
    assert(coeffs.size() == indices.size());
    coeffs_.assign(coeffs.begin(), coeffs.end());
    sparse_index_.assign(indices.begin(), indices.end());
    filter_order_ = *std::max_element(indices.begin(), indices.end()) + 1;

    delay_line_.SetMaximumDelay(filter_order_ + kDefaultBlockSize);
}

float SparseFir::Tick(float in)
{
    delay_line_.Tick(in);

    float y = 0.f;
    for (size_t i = 0; i < coeffs_.size(); ++i)
    {
        uint32_t tap = sparse_index_[i];
        y += coeffs_[i] * delay_line_.TapOut(tap);
    }

    return y;
}

void SparseFir::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1);

    delay_line_.AddNextInputs(input.GetChannelSpan(0));

    std::fill(output.GetChannelSpan(0).begin(), output.GetChannelSpan(0).end(), 0.f);
    delay_line_.GetNextOutputsAt(sparse_index_, output.GetChannelSpan(0), coeffs_);
}

uint32_t SparseFir::InputChannelCount() const
{
    return 1;
}

uint32_t SparseFir::OutputChannelCount() const
{
    return 1;
}

void SparseFir::Clear()
{
    delay_line_.Clear();
}

std::unique_ptr<AudioProcessor> SparseFir::Clone() const
{
    auto clone = std::make_unique<SparseFir>();
    clone->SetCoefficients(coeffs_, sparse_index_);
    return clone;
}

} // namespace sfFDN