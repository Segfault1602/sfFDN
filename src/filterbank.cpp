#include "filterbank.h"

#include <cassert>
#include <mdspan>

namespace fdn
{
FilterBank::FilterBank(size_t filterCount)
{
    filters_.resize(filterCount, nullptr);
    for (size_t i = 0; i < filterCount; i++)
    {
        filters_[i] = new OnePoleFilter();
    }
}

FilterBank::~FilterBank()
{
    for (auto filter : filters_)
    {
        delete filter;
    }
}

void FilterBank::Clear()
{
    for (auto filter : filters_)
    {
        filter->Clear();
    }
}

void FilterBank::SetFilter(size_t index, Filter* filter)
{
    assert(index < filters_.size());
    if (filters_[index] != nullptr)
    {
        delete filters_[index];
    }
    filters_[index] = filter;
}

void FilterBank::Tick(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());

    // Input size must be a multiple of the delay size.
    assert(input.size() % filters_.size() == 0);

    const size_t delay_count = filters_.size();
    const size_t block_size = input.size() / delay_count;

    for (size_t i = 0; i < filters_.size(); ++i)
    {
        auto input_span = input.subspan(i * block_size, block_size);
        auto output_span = output.subspan(i * block_size, block_size);
        filters_[i]->ProcessBlock(input_span.data(), output_span.data(), block_size);
    }

    // auto input_mdspan = std::mdspan(input.data(), block_size, delay_count);
    // auto output_mdspan = std::mdspan(output.data(), block_size, delay_count);

    // for (size_t i = 0; i < input_mdspan.extent(0); i++)
    // {
    //     for (size_t j = 0; j < input_mdspan.extent(1); j++)
    //     {
    //         output_mdspan[i, j] = filters_[j]->Tick(input_mdspan[i, j]);
    //     }
    // }
}

} // namespace fdn