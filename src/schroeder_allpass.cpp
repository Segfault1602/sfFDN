#include "sffdn/schroeder_allpass.h"

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
SchroederAllpass::SchroederAllpass(uint32_t delay, float g)
    : delay_((delay < 1) ? 1 : delay, delay + 1)
    , g_(g)
{
}

void SchroederAllpass::SetDelay(uint32_t delay)
{
    delay = (delay < 1) ? 1 : delay;
    delay_.SetMaximumDelay(delay + 1);
    delay_.SetDelay(delay);
}

void SchroederAllpass::SetG(float g)
{
    g_ = g;
}

float SchroederAllpass::Tick(float input)
{
    float out = delay_.NextOut();
    float v_n = input + (g_ * out);
    delay_.Tick(v_n);
    return out - (g_ * v_n);
}

void SchroederAllpass::ProcessBlock(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == out.size());

    for (uint32_t i = 0; i < in.size(); ++i)
    {
        out[i] = Tick(in[i]);
    }
}

void SchroederAllpass::ProcessBlockAccumulate(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == out.size());

    for (uint32_t i = 0; i < in.size(); ++i)
    {
        out[i] += Tick(in[i]);
    }
}

void SchroederAllpass::Clear()
{
    delay_.Clear();
}

SchroederAllpassSection::SchroederAllpassSection(uint32_t filter_count)
{
    allpasses_.reserve(filter_count);
    for (uint32_t i = 0; i < filter_count; i++)
    {
        allpasses_.emplace_back(1, 0.0f);
    }
}

SchroederAllpassSection::SchroederAllpassSection(SchroederAllpassSection&& other) noexcept
{
    *this = std::move(other);
}

SchroederAllpassSection& SchroederAllpassSection::operator=(SchroederAllpassSection&& other) noexcept
{
    if (this != &other)
    {
        allpasses_ = std::move(other.allpasses_);
    }
    return *this;
}

void SchroederAllpassSection::SetFilterCount(uint32_t filter_count)
{
    allpasses_.resize(filter_count);
}

void SchroederAllpassSection::SetParallel(bool parallel)
{
    parallel_ = parallel;
}

void SchroederAllpassSection::SetDelays(std::span<const uint32_t> delays)
{
    assert(delays.size() == allpasses_.size());
    for (uint32_t i = 0; i < delays.size(); i++)
    {
        allpasses_[i].SetDelay(delays[i]);
    }
}

void SchroederAllpassSection::SetGains(std::span<const float> gains)
{
    assert(gains.size() == allpasses_.size());
    for (uint32_t i = 0; i < gains.size(); i++)
    {
        allpasses_[i].SetG(gains[i]);
    }
}

void SchroederAllpassSection::SetGain(float gain)
{
    for (auto& allpass : allpasses_)
    {
        allpass.SetG(gain);
    }
}

std::vector<uint32_t> SchroederAllpassSection::GetDelays() const
{
    std::vector<uint32_t> delays;
    delays.reserve(allpasses_.size());
    for (const auto& allpass : allpasses_)
    {
        delays.push_back(allpass.GetDelay());
    }
    return delays;
}

std::vector<float> SchroederAllpassSection::GetGains() const
{
    std::vector<float> gains;
    gains.reserve(allpasses_.size());
    for (const auto& allpass : allpasses_)
    {
        gains.push_back(allpass.GetG());
    }
    return gains;
}

void SchroederAllpassSection::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1);

    assert(!allpasses_.empty());

    if (parallel_)
    {
        allpasses_[0].ProcessBlock(input.GetChannelSpan(0), output.GetChannelSpan(0));

        for (auto i = 1u; i < allpasses_.size(); ++i)
        {
            allpasses_[i].ProcessBlockAccumulate(input.GetChannelSpan(0), output.GetChannelSpan(0));
        }
    }
    else
    {
        allpasses_[0].ProcessBlock(input.GetChannelSpan(0), output.GetChannelSpan(0));

        for (auto i = 1u; i < allpasses_.size(); ++i)
        {
            allpasses_[i].ProcessBlock(output.GetChannelSpan(0), output.GetChannelSpan(0));
        }
    }
}

uint32_t SchroederAllpassSection::InputChannelCount() const
{
    return 1;
}

uint32_t SchroederAllpassSection::OutputChannelCount() const
{
    return 1;
}

void SchroederAllpassSection::Clear()
{
    for (auto& allpass : allpasses_)
    {
        allpass.Clear();
    }
}

std::unique_ptr<AudioProcessor> SchroederAllpassSection::Clone() const
{
    auto clone = std::make_unique<SchroederAllpassSection>(allpasses_.size());
    assert(clone->allpasses_.size() == allpasses_.size());
    for (auto i = 0u; i < allpasses_.size(); ++i)
    {
        clone->allpasses_[i].SetDelay(allpasses_[i].GetDelay());
        clone->allpasses_[i].SetG(allpasses_[i].GetG());
    }
    return clone;
}

ParallelSchroederAllpassSection::ParallelSchroederAllpassSection(uint32_t channel_count, uint32_t stage_count)
    : stage_count_(stage_count)
{
    allpasses_.reserve(channel_count);
    for (uint32_t i = 0; i < channel_count; i++)
    {
        allpasses_.emplace_back(stage_count);
    }
}

void ParallelSchroederAllpassSection::SetDelays(std::span<const uint32_t> delays)
{
    assert(delays.size() % allpasses_.size() == 0);
    const uint32_t stage_count = delays.size() / allpasses_.size();

    for (uint32_t i = 0; i < allpasses_.size(); i++)
    {
        auto delay_span = delays.subspan(i * stage_count, stage_count);
        allpasses_[i].SetDelays(delay_span);
    }
}

void ParallelSchroederAllpassSection::SetGains(std::span<const float> gains)
{
    assert(gains.size() == allpasses_.size());
    for (uint32_t i = 0; i < allpasses_.size(); i++)
    {
        allpasses_[i].SetGain(gains[i]);
    }
}

uint32_t ParallelSchroederAllpassSection::InputChannelCount() const
{
    return allpasses_.size();
}

uint32_t ParallelSchroederAllpassSection::OutputChannelCount() const
{
    return allpasses_.size();
}

void ParallelSchroederAllpassSection::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == allpasses_.size());

    for (auto i = 0u; i < allpasses_.size(); ++i)
    {
        auto out_channel_buffer = output.GetChannelBuffer(i);
        allpasses_[i].Process(input.GetChannelBuffer(i), out_channel_buffer);
    }
}

void ParallelSchroederAllpassSection::Clear()
{
    for (auto& allpass : allpasses_)
    {
        allpass.Clear();
    }
}

std::unique_ptr<AudioProcessor> ParallelSchroederAllpassSection::Clone() const
{
    auto clone = std::make_unique<ParallelSchroederAllpassSection>(allpasses_.size(), stage_count_);
    assert(clone->allpasses_.size() == allpasses_.size());
    for (auto i = 0u; i < allpasses_.size(); ++i)
    {
        clone->allpasses_[i].SetDelays(allpasses_[i].GetDelays());
        clone->allpasses_[i].SetGains(allpasses_[i].GetGains());
    }
    return clone;
}

} // namespace sfFDN