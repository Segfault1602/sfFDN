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
DelayBank::DelayBank(const DelayBankConfig& config)
    : block_size_(config.block_size)
    , interpolation_type_(config.interpolation_type)
{
    for (auto delay : config.delays)
    {
        uint32_t max_delay = delay + block_size_;
        if (max_delay % 64 != 0)
        {
            max_delay += 64 - (max_delay % 64);
        }

        delays_.emplace_back(DelayConfig{delay, max_delay, interpolation_type_});
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

std::unique_ptr<AudioProcessor> DelayBank::Clone() const
{
    std::vector<float> delays = GetDelays();

    auto clone = std::make_unique<DelayBank>();
    clone->delays_ = delays_;
    clone->block_size_ = block_size_;
    clone->interpolation_type_ = interpolation_type_;
    return clone;
}

nlohmann::json DelayBank::ToJson() const
{
    nlohmann::json j;
    j["type"] = "DelayBank";
    j["delays"] = GetDelays();
    j["block_size"] = block_size_;
    j["interpolation_type"] = static_cast<uint8_t>(interpolation_type_);
    return j;
}

std::unique_ptr<DelayBank> DelayBank::FromJson(const nlohmann::json& j)
{
    ThrowIfNotType(j, "DelayBank");

    std::vector<float> delays = j["delays"].get<std::vector<float>>();
    uint32_t block_size = j["block_size"].get<uint32_t>();
    DelayInterpolationType interpolation_type = static_cast<DelayInterpolationType>(j.value("interpolation_type", 0));
    return std::make_unique<DelayBank>(DelayBankConfig{delays, block_size, interpolation_type});
}

} // namespace sfFDN