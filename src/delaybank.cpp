#include "sffdn/delaybank.h"

#include <cassert>
#include <cstdint>

namespace sfFDN
{

DelayBank::DelayBank(unsigned long delayCount, unsigned long maxDelay)
{
    for (uint32_t i = 0; i < delayCount; i++)
    {
        delays_.emplace_back(1, maxDelay);
    }
}

DelayBank::DelayBank(std::span<const uint32_t> delays, uint32_t block_size)
{
    for (uint32_t i = 0; i < delays.size(); i++)
    {
        uint32_t max_delay = delays[i] + block_size;
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

uint32_t DelayBank::InputChannelCount() const
{
    return delays_.size();
}

uint32_t DelayBank::OutputChannelCount() const
{
    return delays_.size();
}

void DelayBank::SetDelays(const std::span<const uint32_t> delays, uint32_t block_size)
{
    assert(delays.size() == delays_.size());
    for (uint32_t i = 0; i < delays.size(); i++)
    {
        delays_[i].SetMaximumDelay(delays[i] + block_size);
        delays_[i].SetDelay(delays[i]);
    }
}

std::vector<uint32_t> DelayBank::GetDelays() const
{
    std::vector<uint32_t> delays;
    delays.reserve(delays_.size());
    for (const auto& delay : delays_)
    {
        delays.push_back(delay.GetDelay());
    }
    return delays;
}

void DelayBank::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delays_.size());

    for (uint32_t i = 0; i < delays_.size(); i++)
    {
        auto output_buffer = output.GetChannelBuffer(i);
        delays_[i].Process(input.GetChannelBuffer(i), output_buffer);
    }
}

void DelayBank::AddNextInputs(const AudioBuffer& input)
{
    assert(input.ChannelCount() == delays_.size());

    for (uint32_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].AddNextInputs(input.GetChannelSpan(i));
    }
}

void DelayBank::GetNextOutputs(AudioBuffer& output)
{
    assert(output.ChannelCount() == delays_.size());

    for (uint32_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].GetNextOutputs(output.GetChannelSpan(i));
    }
}

} // namespace sfFDN