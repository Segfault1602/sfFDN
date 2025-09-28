#include "sffdn/filterbank.h"
#include "sffdn/audio_buffer.h"

#include "pch.h"

namespace sfFDN
{
FilterBank::FilterBank() = default;

void FilterBank::Clear()
{
    for (auto& filter : filters_)
    {
        filter->Clear();
    }
}

void FilterBank::AddFilter(std::unique_ptr<AudioProcessor> filter)
{
    filters_.push_back(std::move(filter));
}

void FilterBank::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == filters_.size());

    for (auto i = 0u; i < filters_.size(); ++i)
    {
        auto input_buf = input.GetChannelBuffer(i);
        auto output_buf = output.GetChannelBuffer(i);
        filters_[i]->Process(input_buf, output_buf);
    }
}

uint32_t FilterBank::InputChannelCount() const
{
    return filters_.size();
}

uint32_t FilterBank::OutputChannelCount() const
{
    return filters_.size();
}

std::unique_ptr<AudioProcessor> FilterBank::Clone() const
{
    auto clone = std::make_unique<FilterBank>();
    for (const auto& filter : filters_)
    {
        clone->AddFilter(filter->Clone());
    }
    return clone;
}

} // namespace sfFDN