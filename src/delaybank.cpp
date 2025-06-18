#include "delaybank.h"

#include <cassert>
#include <mdspan>

namespace fdn
{

DelayBank::DelayBank(unsigned long delayCount, unsigned long maxDelay)
{
    for (size_t i = 0; i < delayCount; i++)
    {
        delays_.emplace_back(1, maxDelay);
    }
}

DelayBank::DelayBank(std::span<const size_t> delays, size_t block_size)
{
    for (size_t i = 0; i < delays.size(); i++)
    {
        size_t max_delay = delays[i] + block_size;
        if (max_delay % 64 != 0)
        {
            max_delay += 64 - (max_delay % 64);
        }

        delays_.emplace_back(delays[i], max_delay);
    }
}

void DelayBank::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

size_t DelayBank::InputChannelCount() const
{
    return delays_.size();
}

size_t DelayBank::OutputChannelCount() const
{
    return delays_.size();
}

void DelayBank::SetDelays(const std::span<const size_t> delays)
{
    assert(delays.size() == delays_.size());
    for (size_t i = 0; i < delays.size(); i++)
    {
        delays_[i].SetMaximumDelay(delays[i] + 512);
        delays_[i].SetDelay(delays[i]);
    }
}

void DelayBank::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delays_.size());

    AddNextInputs(input);
    GetNextOutputs(output);
}

void DelayBank::AddNextInputs(const AudioBuffer& input)
{
    assert(input.ChannelCount() == delays_.size());

    for (size_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].AddNextInputs(input.GetChannelSpan(i));
    }
}

void DelayBank::GetNextOutputs(AudioBuffer& output)
{
    assert(output.ChannelCount() == delays_.size());

    for (size_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].GetNextOutputs(output.GetChannelSpan(i));
    }
}

} // namespace fdn