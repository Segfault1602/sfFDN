#include "sffdn/delaybank.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace sfFDN
{

DelayBank::DelayBank(std::span<const uint32_t> delays, uint32_t block_size)
{
    for (unsigned int delay : delays)
    {
        uint32_t max_delay = delay + block_size;
        if (max_delay % 64 != 0)
        {
            max_delay += 64 - (max_delay % 64);
        }

        delays_.emplace_back(delay, max_delay);
    }
}

DelayBank::DelayBank(const DelayBank& other)
    : delays_(other.delays_)
{
}

DelayBank& DelayBank::operator=(const DelayBank& other)
{
    if (this != &other)
    {
        delays_ = other.delays_;
    }
    return *this;
}

DelayBank::DelayBank(DelayBank&& other) noexcept
    : delays_(std::move(other.delays_))
{
}

DelayBank& DelayBank::operator=(DelayBank&& other) noexcept
{
    delays_ = std::move(other.delays_);
    return *this;
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
    delays_.resize(delays.size());
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

void DelayBank::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
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
    if (input.SampleCount() > 1)
    {
        for (uint32_t i = 0; i < delays_.size(); i++)
        {
            delays_[i].AddNextInputs(input.GetChannelSpan(i));
        }
    }
    else
    {
        for (uint32_t i = 0; i < delays_.size(); i++)
        {
            delays_[i].Tick(input.GetChannelSpan(i)[0]);
        }
    }
}

void DelayBank::GetNextOutputs(AudioBuffer& output)
{
    assert(output.ChannelCount() == delays_.size());

    if (output.SampleCount() > 1)
    {
        for (uint32_t i = 0; i < delays_.size(); i++)
        {
            delays_[i].GetNextOutputs(output.GetChannelSpan(i));
        }
    }
    else
    {
        for (uint32_t i = 0; i < delays_.size(); i++)
        {
            output.GetChannelSpan(i)[0] = delays_[i].NextOut();
        }
    }
}

std::unique_ptr<AudioProcessor> DelayBank::Clone() const
{
    auto clone = std::make_unique<DelayBank>(*this);
    return clone;
}

} // namespace sfFDN