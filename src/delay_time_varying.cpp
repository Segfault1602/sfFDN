#include "delay_time_varying.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace sfFDN
{

DelayTimeVarying::DelayTimeVarying(float delay, uint32_t max_delay, uint32_t sample_rate)
    : delayA_(delay, max_delay)
    , delay_(delay)
    , samplerate_(sample_rate)
    , mod_freq_(0)
    , mod_depth_(0)
    , mod_phase_(0)
    , mod_phase_increment_(0)
{
}

void DelayTimeVarying::Clear()
{
    delayA_.Clear();
}

void DelayTimeVarying::SetMaximumDelay(uint32_t delay)
{
    delayA_.SetMaximumDelay(delay);
}

void DelayTimeVarying::SetDelay(float delay)
{
    delay_ = delay;
    delayA_.SetDelay(delay);
}

void DelayTimeVarying::SetMod(float freq, float depth_ms)
{
    mod_freq_ = freq;
    mod_depth_ = samplerate_ * depth_ms / 1000;
    mod_phase_increment_ = 2 * std::numbers::pi_v<float> * freq / samplerate_;
}

void DelayTimeVarying::UpdateDelay()
{
    mod_phase_ += mod_phase_increment_;
    if (mod_phase_ > 2 * std::numbers::pi_v<float>)
    {
        mod_phase_ -= 2 * std::numbers::pi_v<float>;
    }

    float mod = mod_depth_ * std::sin(mod_phase_);
    delayA_.SetDelay(delay_ + mod);
}

float DelayTimeVarying::Tick(float input)
{
    // UpdateDelay();

    return delayA_.Tick(input);
}

void DelayTimeVarying::AddNextInput(float input)
{
    delayA_.Tick(input);
}

} // namespace sfFDN