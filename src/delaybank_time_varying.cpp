#include "sffdn/delaybank_time_varying.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sfFDN
{

DelayBankTimeVarying::DelayBankTimeVarying(std::span<const float> delays, uint32_t max_delay,
                                           DelayInterpolationType type)
{
    for (auto delay : delays)
    {
        delays_.emplace_back(delay, max_delay, type);
    }
}

DelayBankTimeVarying::DelayBankTimeVarying(const DelayBankTimeVarying& other)
    : delays_(other.delays_)
{
}

DelayBankTimeVarying& DelayBankTimeVarying::operator=(const DelayBankTimeVarying& other)
{
    if (this != &other)
    {
        delays_ = other.delays_;
    }
    return *this;
}

DelayBankTimeVarying::DelayBankTimeVarying(DelayBankTimeVarying&& other) noexcept
    : delays_(std::move(other.delays_))
{
}

DelayBankTimeVarying& DelayBankTimeVarying::operator=(DelayBankTimeVarying&& other) noexcept
{
    delays_ = std::move(other.delays_);
    return *this;
}

void DelayBankTimeVarying::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

uint32_t DelayBankTimeVarying::InputChannelCount() const
{
    return delays_.size();
}

uint32_t DelayBankTimeVarying::OutputChannelCount() const
{
    return delays_.size();
}

void DelayBankTimeVarying::SetDelays(const std::span<const uint32_t> delays, uint32_t block_size)
{
    delays_.resize(delays.size());
    for (uint32_t i = 0; i < delays.size(); i++)
    {
        delays_[i].SetMaximumDelay(delays[i] + block_size);
        delays_[i].SetDelay(delays[i]);
    }
}

void DelayBankTimeVarying::SetMods(const std::span<const float> freqs, const std::span<const float> depths,
                                   const std::span<const float> phase_offsets)
{
    if (freqs.size() != delays_.size() || depths.size() != delays_.size() ||
        (!phase_offsets.empty() && phase_offsets.size() != delays_.size()))
    {
        throw std::invalid_argument("SetMods: size of freqs, depths, and phase_offsets must match number of delays");
    }

    for (uint32_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].SetMod(freqs[i], depths[i], phase_offsets.empty() ? 0.0f : phase_offsets[i]);
    }
}

std::vector<uint32_t> DelayBankTimeVarying::GetDelays() const
{
    std::vector<uint32_t> delays;
    delays.reserve(delays_.size());
    for (const auto& delay : delays_)
    {
        delays.push_back(delay.GetDelay());
    }
    return delays;
}

void DelayBankTimeVarying::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
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

std::unique_ptr<AudioProcessor> DelayBankTimeVarying::Clone() const
{
    auto clone = std::make_unique<DelayBankTimeVarying>(*this);
    return clone;
}

} // namespace sfFDN