#include "delay_time_varying.h"

#include <cmath>
#include <numbers>

namespace sfFDN
{

DelayTimeVarying::DelayTimeVarying(float delay, unsigned long maxDelay, size_t samplerate)
    : delayA_(delay, maxDelay)
    , delay_(delay)
    , samplerate_(samplerate)
    , mod_freq_(0)
    , mod_depth_(0)
    , mod_phase_(0)
    , mod_phase_increment_(0)
{
}

void DelayTimeVarying::Clear(void)
{
    delayA_.Clear();
}

void DelayTimeVarying::SetMaximumDelay(unsigned long delay)
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

    // return delayA_.Tick(input);
    return 0.0f; // Placeholder return value
}

void DelayTimeVarying::AddNextInput(float input)
{
    // delayA_.AddNextInput(input);
}

float DelayTimeVarying::GetNextOutput()
{
    // UpdateDelay();

    // return delayA_.GetNextOutput();
    return 0.0f; // Placeholder return value
}

} // namespace sfFDN