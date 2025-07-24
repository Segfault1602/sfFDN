#include "delay_matrix.h"

#include <cassert>
#include <iostream>

namespace sfFDN
{

DelayMatrix::DelayMatrix(uint32_t N, std::span<const uint32_t> delays)
    : FeedbackMatrix(N)
    , delays_(N)
{
    delays_.SetDelays(delays);
}

void DelayMatrix::Clear()
{
    delays_.Clear();
}

void DelayMatrix::SetDelays(std::span<uint32_t> delays)
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

void DelayMatrix::PrintInfo() const
{
    std::cout << "DelayMatrix Info:" << std::endl;
    std::cout << "Delays: [";
    auto delays = delays_.GetDelays();
    for (size_t i = 0; i < delays.size(); ++i)
    {
        std::cout << delays[i];
        if (i < delays.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Mixing Matrix: ";
    mixing_matrix_.Print();
}

} // namespace sfFDN