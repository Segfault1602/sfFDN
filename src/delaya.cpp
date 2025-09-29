#include "sffdn/delaya.h"

#include "sffdn/audio_buffer.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <print>

namespace sfFDN
{

DelayAllpass::DelayAllpass(float delay, uint32_t max_delay)
    : delay_(static_cast<uint32_t>(delay + 1), max_delay)
{
    if (delay < 0.5f)
    {
        std::println(std::cerr, "DelayAllpass::DelayAllpass: delay must be >= 0.5!");
        assert(false);
        delay = 0.5f; // Set to minimum valid value
    }

    this->SetDelay(delay);
}

void DelayAllpass::Clear()
{
    delay_.Clear();
    allpass_.Clear();
}

void DelayAllpass::SetMaximumDelay(uint32_t delay)
{
    delay_.SetMaximumDelay(delay);
}

void DelayAllpass::SetDelay(float delay)
{
    if (delay < 0.5)
    {
        std::println(std::cerr, "DelayAllpass::setDelay: argument ({}) less than 0.5 not possible!", delay);
        assert(false);
        return;
    }

    int int_delay = static_cast<int>(delay);
    float alpha = delay - int_delay;

    if (alpha < 0.5f)
    {
        int_delay -= 1;
        alpha += 1.0f;
    }

    assert(int_delay >= 0);
    delay_.SetDelay(int_delay);

    allpass_.SetCoefficients((1.0f - alpha) / (1.0f + alpha));
}

float DelayAllpass::Tick(float input)
{
    float out = delay_.Tick(input);
    return allpass_.Tick(out);
}

void DelayAllpass::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    delay_.Process(input, output);
    allpass_.Process(output, output);
}

} // namespace sfFDN