#include "sffdn/delay_interp.h"

#include "sffdn/audio_buffer.h"

#include <array>
#include <cassert>
#include <cstdint>

namespace sfFDN
{

DelayInterp::DelayInterp(float delay, uint32_t max_delay, DelayInterpolationType type)
    : delayline_(static_cast<uint32_t>(delay + 1), max_delay)
    , type_(type)
    , delay_(0)
    , int_delay_(0)
    , frac_delay_(0.0f)
{
    this->SetDelay(delay);
}

void DelayInterp::Clear()
{
    delayline_.Clear();
    allpass_.Clear();
}

void DelayInterp::SetMaximumDelay(uint32_t delay)
{
    delayline_.SetMaximumDelay(delay);
}

void DelayInterp::SetDelay(float delay)
{
    delay_ = delay;
    int_delay_ = static_cast<uint32_t>(delay);
    frac_delay_ = delay - static_cast<float>(int_delay_);

    if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.SetDelay(int_delay_);
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        if (frac_delay_ < 0.5f)
        {
            int_delay_ -= 1;
            frac_delay_ += 1.0f;
        }

        assert(int_delay_ >= 0);

        const bool update_allpass = delayline_.GetDelay() != int_delay_;

        delayline_.SetDelay(int_delay_);

        allpass_.SetCoefficients((1.0f - frac_delay_) / (1.0f + frac_delay_));
        if (update_allpass)
        {
            allpass_.Tick(delayline_.LastOut());
        }
    }
}

float DelayInterp::Tick(float input)
{
    switch (type_)
    {
    case DelayInterpolationType::Linear:
    {
        delayline_.Tick(input);
        const float a = delayline_.TapOut(int_delay_);
        const float b = delayline_.TapOut(int_delay_ + 1);
        return a + (b - a) * frac_delay_;
    }
    case DelayInterpolationType::Allpass:
    {
        const float out = delayline_.Tick(input);
        return allpass_.Tick(out);
    }
    }

    assert(false);
    return 0.0f;
}

void DelayInterp::ProcessLinear(const AudioBuffer& input, AudioBuffer& output)
{
    auto in_span = input.GetChannelSpan(0);
    auto out_span = output.GetChannelSpan(0);

    if (delayline_.AddNextInputs(in_span))
    {
        std::array<uint32_t, 2> taps = {int_delay_, int_delay_ + 1};
        std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        delayline_.GetNextOutputsAt(taps, out_span, coeffs);
    }
    else
    {
        for (uint32_t n = 0; n < input.SampleCount(); ++n)
        {
            out_span[n] = this->Tick(in_span[n]);
        }
    }
}

void DelayInterp::ProcessAllpass(const AudioBuffer& input, AudioBuffer& output)
{
    delayline_.Process(input, output);
    allpass_.Process(output, output);
}

void DelayInterp::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    if (type_ == DelayInterpolationType::Linear)
    {
        this->ProcessLinear(input, output);
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        this->ProcessAllpass(input, output);
    }
}

} // namespace sfFDN