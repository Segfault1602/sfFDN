#include "parallel_gains.h"

#include <cassert>

#include "array_math.h"

namespace sfFDN
{

ParallelGains::ParallelGains(ParallelGainsMode mode)
    : gains_(1, 1.0f) // Default to one channel with gain of 1.0
    , mode_(mode)
{
}

ParallelGains::ParallelGains(size_t N, ParallelGainsMode mode, float gain)
    : gains_(N, gain)
    , mode_(mode)
{
}

ParallelGains::ParallelGains(ParallelGainsMode mode, std::span<const float> gains)
    : mode_(mode)
{
    SetGains(gains);
}

void ParallelGains::SetMode(ParallelGainsMode mode)
{
    mode_ = mode;
}

void ParallelGains::SetGains(std::span<const float> gains)
{
    assert(gains.size() > 0);
    gains_.assign(gains.begin(), gains.end());
}

size_t ParallelGains::InputChannelCount() const
{
    return mode_ == ParallelGainsMode::Multiplexed ? 1 : gains_.size();
}

size_t ParallelGains::OutputChannelCount() const
{
    return mode_ == ParallelGainsMode::Multiplexed ? gains_.size() : 1;
}

void ParallelGains::Process(const AudioBuffer& input, AudioBuffer& output)
{
    if (mode_ == ParallelGainsMode::Multiplexed)
    {
        ProcessBlockMultiplexed(input, output);
    }
    else
    {
        assert(mode_ == ParallelGainsMode::DeMultiplexed);
        ProcessBlockDeMultiplexed(input, output);
    }
}

void ParallelGains::ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == gains_.size());
    assert(input.SampleCount() == output.SampleCount());

    for (size_t i = 0; i < gains_.size(); i++)
    {
        ArrayMath::Scale(input.GetChannelSpan(0), gains_[i], output.GetChannelSpan(i));
    }
}

void ParallelGains::ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == gains_.size());
    assert(output.ChannelCount() == 1);

    for (size_t i = 0; i < gains_.size(); i++)
    {
        ArrayMath::ScaleAccumulate(input.GetChannelSpan(i), gains_[i], output.GetChannelSpan(0));
    }
}

} // namespace sfFDN