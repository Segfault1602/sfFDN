#include "sffdn/filterbank.h"

#include <cassert>

namespace sfFDN
{
FilterBank::FilterBank() = default;

FilterBank::~FilterBank() = default;

void FilterBank::Clear()
{
    // for (auto filter : filters_)
    // {
    //     filter->Clear();
    // }
}

void FilterBank::AddFilter(std::unique_ptr<AudioProcessor> filter)
{
    filters_.push_back(std::move(filter));
}

void FilterBank::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == filters_.size());

    for (auto i = 0; i < filters_.size(); ++i)
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

} // namespace sfFDN