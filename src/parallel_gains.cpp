#include <algorithm>

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/parallel_gains.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "array_math.h"

namespace sfFDN
{

ParallelGains::ParallelGains(ParallelGainsMode mode)
    : gains_(1, 1.0f) // Default to one channel with gain of 1.0
    , mode_(mode)
{
}

ParallelGains::ParallelGains(uint32_t channel_count, ParallelGainsMode mode, float gain)
    : gains_(channel_count, gain)
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
    assert(!gains.empty());
    gains_.assign(gains.begin(), gains.end());
}

void ParallelGains::GetGains(std::span<float> gains) const
{
    assert(gains.size() == gains_.size());
    std::ranges::copy(gains_, gains.begin());
}

uint32_t ParallelGains::InputChannelCount() const
{
    switch (mode_)
    {
    case ParallelGainsMode::Split:
        return 1; // Single input channel for multiplexed mode
    case ParallelGainsMode::Merge:
    case ParallelGainsMode::Parallel:
        return gains_.size(); // One input channel per gain in de-multiplexed and parallel modes
    default:
        assert(false && "Unknown ParallelGainsMode");
        return 0; // Should never reach here
    }
}

uint32_t ParallelGains::OutputChannelCount() const
{
    switch (mode_)
    {
    case ParallelGainsMode::Split:
        return gains_.size(); // One output channel per gain in multiplexed mode
    case ParallelGainsMode::Merge:
        return 1; // Single output channel for de-multiplexed mode
    case ParallelGainsMode::Parallel:
        return gains_.size(); // One output channel per gain in parallel mode
    default:
        assert(false && "Unknown ParallelGainsMode");
        return 0; // Should never reach here
    }
}

void ParallelGains::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    if (mode_ == ParallelGainsMode::Split)
    {
        ProcessBlockMultiplexed(input, output);
    }
    else if (mode_ == ParallelGainsMode::Merge)
    {
        ProcessBlockDeMultiplexed(input, output);
    }
    else if (mode_ == ParallelGainsMode::Parallel)
    {
        ProcessBlockParallel(input, output);
    }
    else
    {
        assert(false && "Unknown ParallelGainsMode");
    }
}

void ParallelGains::ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == gains_.size());
    assert(input.SampleCount() == output.SampleCount());

    for (auto i = 0u; i < gains_.size(); i++)
    {
        ArrayMath::Scale(input.GetChannelSpan(0), gains_[i], output.GetChannelSpan(i));
    }
}

void ParallelGains::ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == gains_.size());
    assert(output.ChannelCount() == 1);

    for (auto i = 0u; i < gains_.size(); i++)
    {
        ArrayMath::ScaleAccumulate(input.GetChannelSpan(i), gains_[i], output.GetChannelSpan(0));
    }
}

void ParallelGains::ProcessBlockParallel(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == gains_.size());
    assert(output.ChannelCount() == gains_.size());

    for (auto i = 0u; i < gains_.size(); i++)
    {
        ArrayMath::Scale(input.GetChannelSpan(i), gains_[i], output.GetChannelSpan(i));
    }
}

void ParallelGains::Clear()
{
    // no-op
}

std::unique_ptr<AudioProcessor> ParallelGains::Clone() const
{
    return std::make_unique<ParallelGains>(mode_, gains_);
}

} // namespace sfFDN