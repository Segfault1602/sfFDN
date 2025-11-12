#include "sffdn/delay_time_varying.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

namespace sfFDN
{

DelayTimeVarying::DelayTimeVarying(float delay, uint32_t max_delay, DelayInterpolationType type)
    : delay_(delay, max_delay, type)
    , base_delay_(delay)
    , lfo_(0.0f, 0.0f)
{
}

void DelayTimeVarying::Clear()
{
    delay_.Clear();
    lfo_.ResetPhase();
}

void DelayTimeVarying::SetMaximumDelay(uint32_t delay)
{
    delay_.SetMaximumDelay(delay);
}

void DelayTimeVarying::SetDelay(float delay)
{
    delay_.SetDelay(delay);
    base_delay_ = delay;
}

float DelayTimeVarying::GetDelay() const
{
    return delay_.GetDelay();
}

void DelayTimeVarying::SetMod(float freq, float amp, float phase_offset)
{
    if (delay_.GetDelay() < amp)
    {
        throw std::invalid_argument("SetMod: depth must be less than the current delay");
    }

    if (delay_.GetDelay() + amp > delay_.GetMaximumDelay())
    {
        throw std::invalid_argument("SetMod: depth + base delay must be less than the maximum delay");
    }

    lfo_.SetFrequency(freq);
    lfo_.SetAmplitude(amp);
    lfo_.SetPhaseOffset(phase_offset);
}

void DelayTimeVarying::UpdateDelay()
{
    float mod = lfo_.Tick();
    delay_.SetDelay(base_delay_ + mod);
}

float DelayTimeVarying::Tick(float input)
{
    UpdateDelay();

    return delay_.Tick(input);
}

void DelayTimeVarying::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    auto in_span = input.GetChannelSpan(0);
    auto out_span = output.GetChannelSpan(0);

    constexpr uint32_t kUnrollFactor = 16;
    const uint32_t size = in_span.size();
    const uint32_t unroll_size = size & ~(kUnrollFactor - 1);

    uint32_t sample = 0;
    for (; sample < unroll_size; sample += kUnrollFactor)
    {
        std::array<float, kUnrollFactor> mods{};
        lfo_.Generate(mods);

        auto in_batch = in_span.subspan(sample, kUnrollFactor);
        auto out_batch = out_span.subspan(sample, kUnrollFactor);

        for (auto i = 0u; i < kUnrollFactor; ++i)
        {
            delay_.SetDelay(base_delay_ + mods[i]);
            out_batch[i] = delay_.Tick(in_batch[i]);
        }
    }

    for (; sample < size; ++sample)
    {
        UpdateDelay();
        out_span[sample] = delay_.Tick(in_span[sample]);
    }
}

} // namespace sfFDN