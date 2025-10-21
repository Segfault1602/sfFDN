#include "sffdn/parallel_gains.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sfFDN
{
TimeVaryingParallelGains::TimeVaryingParallelGains(ParallelGainsMode mode)
    : mode_(mode)
{
    lfos_.emplace_back(0.0f, 0.0f); // Default to one LFO with 0 Hz
    lfos_[0].SetAmplitude(0.0f);
    lfos_[0].SetOffset(1.0f);
}

TimeVaryingParallelGains::TimeVaryingParallelGains(uint32_t channel_count, ParallelGainsMode mode, float gain)
    : mode_(mode)
{
    lfos_.reserve(channel_count);
    for (uint32_t i = 0; i < channel_count; ++i)
    {
        lfos_.emplace_back(0.0f, 0.0f);
        lfos_[i].SetAmplitude(0.0f);
        lfos_[i].SetOffset(gain);
    }
}

TimeVaryingParallelGains::TimeVaryingParallelGains(ParallelGainsMode mode, std::span<const float> gains)
    : mode_(mode)
{
    lfos_.reserve(gains.size());
    for (const float& gain : gains)
    {
        lfos_.emplace_back(0.0f, 0.0f);
        lfos_.back().SetAmplitude(0.0f);
        lfos_.back().SetOffset(gain);
    }
}

void TimeVaryingParallelGains::SetCenterGains(std::span<const float> gains)
{
    assert(!gains.empty());
    lfos_.resize(gains.size());
    for (auto i = 0u; i < gains.size(); ++i)
    {
        lfos_[i].SetOffset(gains[i]);
    }
}

void TimeVaryingParallelGains::GetCenterGains(std::span<float> gains) const
{
    assert(gains.size() == lfos_.size());
    for (auto i = 0u; i < lfos_.size(); ++i)
    {
        gains[i] = lfos_[i].GetOffset();
    }
}

void TimeVaryingParallelGains::SetLfoFrequency(std::span<const float> frequencies)
{
    assert(!frequencies.empty());
    assert(frequencies.size() == lfos_.size());

    lfos_.resize(frequencies.size());

    for (auto i = 0u; i < frequencies.size(); ++i)
    {
        lfos_[i].SetFrequency(frequencies[i]);
    }
}

void TimeVaryingParallelGains::SetLfoAmplitude(std::span<const float> amplitudes)
{
    assert(!amplitudes.empty());
    assert(amplitudes.size() == lfos_.size());

    lfos_.resize(amplitudes.size());

    for (auto i = 0u; i < amplitudes.size(); ++i)
    {
        lfos_[i].SetAmplitude(amplitudes[i]);
    }
}

void TimeVaryingParallelGains::SetLfoPhaseOffset(std::span<const float> phase_offsets)
{
    assert(!phase_offsets.empty());
    assert(phase_offsets.size() == lfos_.size());

    lfos_.resize(phase_offsets.size());

    for (auto i = 0u; i < phase_offsets.size(); ++i)
    {
        lfos_[i].SetPhaseOffset(phase_offsets[i]);
    }
}

uint32_t TimeVaryingParallelGains::InputChannelCount() const
{
    switch (mode_)
    {
    case ParallelGainsMode::Split:
        return 1; // Single input channel for multiplexed mode
    case ParallelGainsMode::Merge:
    case ParallelGainsMode::Parallel:
        return lfos_.size(); // One input channel per gain in de-multiplexed and parallel modes
    default:
        assert(false && "Unknown ParallelGainsMode");
        return 0; // Should never reach here
    }
}

uint32_t TimeVaryingParallelGains::OutputChannelCount() const
{
    switch (mode_)
    {
    case ParallelGainsMode::Split:
        return lfos_.size(); // One output channel per gain in multiplexed mode
    case ParallelGainsMode::Merge:
        return 1; // Single output channel for de-multiplexed mode
    case ParallelGainsMode::Parallel:
        return lfos_.size(); // One output channel per gain in parallel mode
    default:
        assert(false && "Unknown ParallelGainsMode");
        return 0; // Should never reach here
    }
}

void TimeVaryingParallelGains::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
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
        assert(false && "Unknown TimeVaryingParallelGains");
    }
}

void TimeVaryingParallelGains::ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == lfos_.size());
    assert(input.SampleCount() == output.SampleCount());

    for (auto i = 0u; i < lfos_.size(); i++)
    {
        lfos_[i].Multiply(input.GetChannelSpan(0), output.GetChannelSpan(i));
    }
}

void TimeVaryingParallelGains::ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == lfos_.size());
    assert(output.ChannelCount() == 1);

    for (auto i = 0u; i < lfos_.size(); i++)
    {
        lfos_[i].MultiplyAccumulate(input.GetChannelSpan(i), output.GetChannelSpan(0));
    }
}

void TimeVaryingParallelGains::ProcessBlockParallel(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == lfos_.size());
    assert(output.ChannelCount() == lfos_.size());

    for (auto i = 0u; i < lfos_.size(); i++)
    {
        lfos_[i].Multiply(input.GetChannelSpan(i), output.GetChannelSpan(i));
    }
}

void TimeVaryingParallelGains::Clear()
{
    for (auto& lfo : lfos_)
    {
        lfo.ResetPhase();
    }
}

std::unique_ptr<AudioProcessor> TimeVaryingParallelGains::Clone() const
{
    auto clone = std::make_unique<TimeVaryingParallelGains>(mode_);
    clone->lfos_ = lfos_;

    return clone;
}

} // namespace sfFDN