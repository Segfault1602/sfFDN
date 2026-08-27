#include "sffdn/delaybank.h"

#include "json_helper.h"
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
DelayBank::DelayBank(const DelayBankOptions& config)
    : block_size_(config.block_size)
    , interpolation_type_(config.interpolation_type)
{
    for (auto delay : config.delays)
    {
        uint32_t max_delay = delay + block_size_ * 2;
        if (max_delay % 64 != 0)
        {
            max_delay += 64 - (max_delay % 64);
        }

        delays_.emplace_back(DelayOptions{.delay = delay, .max_delay = max_delay, .interp_type = interpolation_type_});
    }
}

void DelayBank::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

uint32_t DelayBank::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return delays_.size();
}

uint32_t DelayBank::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return delays_.size();
}

void DelayBank::SetDelays(const std::span<const float> delays, uint32_t block_size)
{
    block_size_ = block_size;
    delays_.resize(delays.size());
    for (uint32_t i = 0; i < delays.size(); i++)
    {
        delays_[i].SetMaximumDelay(delays[i] + block_size);
        delays_[i].SetDelay(delays[i]);
    }
}

std::vector<float> DelayBank::GetDelays() const
{
    std::vector<float> delays;
    delays.reserve(delays_.size());
    for (const auto& delay : delays_)
    {
        delays.push_back(delay.GetDelay());
    }
    return delays;
}

void DelayBank::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delays_.size());

    uint32_t channel = 0;
    for (auto& delay : delays_)
    {
        auto output_buffer = output.GetChannelBuffer(channel);
        delay.Process(input.GetChannelBuffer(channel), output_buffer);
        ++channel;
    }
}

void DelayBank::AddNextInputs(const AudioBuffer& input) noexcept SFFDN_NONBLOCKING
{
    assert(input.ChannelCount() == delays_.size());
    uint32_t channel = 0;
    for (auto& delay : delays_)
    {
        delay.AddNextInputs(input.GetChannelSpan(channel));
        ++channel;
    }
}

void DelayBank::GetNextOutputs(AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(output.ChannelCount() == delays_.size());

    uint32_t channel = 0;
    for (auto& delay : delays_)
    {
        delay.GetNextOutputs(output.GetChannelSpan(channel));
        ++channel;
    }
}

std::unique_ptr<AudioProcessor> DelayBank::Clone() const
{
    auto clone = std::make_unique<DelayBank>();
    clone->delays_ = delays_;
    clone->block_size_ = block_size_;
    clone->interpolation_type_ = interpolation_type_;
    return clone;
}

} // namespace sfFDN