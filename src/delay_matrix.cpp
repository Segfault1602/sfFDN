#include "delay_matrix.h"

#include <cassert>
#include <iostream>

namespace sfFDN
{

DelayMatrix::DelayMatrix(size_t N, std::span<const size_t> delays)
    : FeedbackMatrix(N)
    , delays_(N)
{
}

void DelayMatrix::Clear()
{
    delays_.Clear();
}

void DelayMatrix::SetDelays(std::span<size_t> delays)
{
    delays_.SetDelays(delays);
}

void DelayMatrix::SetMatrix(ScalarFeedbackMatrix mixing_matrix)
{
    assert(mixing_matrix.GetSize() == N_);
    mixing_matrix_ = mixing_matrix;
}

void DelayMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delays_.InputChannelCount());

    mixing_matrix_.Process(input, output);

    delays_.Process(output, output);
}

} // namespace sfFDN