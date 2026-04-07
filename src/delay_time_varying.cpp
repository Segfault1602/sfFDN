#include "sffdn/delay_time_varying.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/delay_interp.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace sfFDN
{
template <DelayInterpolationType type>
DelayTimeVarying<type>::DelayTimeVarying(float delay, uint32_t max_delay)
    : delay_(delay, max_delay)
    , base_delay_(delay)
    , lfo_(0.0f, 0.0f)
{
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::Clear()
{
    delay_.Clear();
    lfo_.ResetPhase();
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::SetMaximumDelay(uint32_t delay)
{
    delay_.SetMaximumDelay(delay);
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::SetDelay(float delay)
{
    delay_.SetDelay(delay);
    base_delay_ = delay;
}

template <DelayInterpolationType type>
float DelayTimeVarying<type>::GetDelay() const
{
    return delay_.GetDelay();
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::SetMod(float freq, float amplitude, float phase_offset)
{
    if (delay_.GetDelay() < amplitude)
    {
        throw std::invalid_argument("SetMod: amplitude must be less than the current delay");
    }

    if (delay_.GetDelay() + amplitude > delay_.GetMaximumDelay())
    {
        throw std::invalid_argument("SetMod: amplitude + base delay must be less than the maximum delay");
    }

    lfo_.SetFrequency(freq);
    lfo_.SetAmplitude(amplitude);
    lfo_.SetPhaseOffset(phase_offset);
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::UpdateDelay()
{
    delay_.SetDelay(base_delay_ + lfo_.Tick());
}

template <DelayInterpolationType type>
float DelayTimeVarying<type>::Tick(float input)
{
    UpdateDelay();

    return delay_.Tick(input);
}

template <DelayInterpolationType type>
void DelayTimeVarying<type>::Process(const AudioBuffer& input, AudioBuffer& output)
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

template <DelayInterpolationType type>
nlohmann::json DelayTimeVarying<type>::ToJson() const
{
    nlohmann::json j;
    j["type"] = "DelayTimeVarying";
    j["delay"] = delay_.ToJson();
    j["base_delay"] = base_delay_;
    j["lfo"] = lfo_.ToJson();
    return j;
}

template class DelayTimeVarying<DelayInterpolationType::Linear>;
template class DelayTimeVarying<DelayInterpolationType::Allpass>;

} // namespace sfFDN