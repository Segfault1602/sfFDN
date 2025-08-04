#include "sffdn/filter.h"

#include <cmath>
#include <iostream>
#include <numbers>

namespace
{
constexpr float TWO_PI = std::numbers::pi_v<float> * 2;
}

namespace sfFDN
{

OnePoleFilter::OnePoleFilter()
    : gain_(1.0f)
    , b0_(1.0f)
    , a1_(0.0f)
    , state_{0.0f, 0.0f}
{
}

void OnePoleFilter::SetPole(float pole)
{
    // https://ccrma.stanford.edu/~jos/fp/One_Pole.html
    // If the filter has a pole at z = -a, then a_[1] will be -pole;
    assert(pole <= 1.f && pole >= -1.f);

    // Set the b value to 1 - |a| to get a peak gain of 1.
    b0_ = 1.f - std::abs(pole);
    a1_ = -pole;
}

void OnePoleFilter::SetCoefficients(float b0, float a1)
{
    b0_ = b0;
    a1_ = a1;
}

void OnePoleFilter::SetDecayFilter(float decayDb, float timeMs, float samplerate)
{
    assert(decayDb < 0.f);
    const float lambda = std::log(std::pow(10.f, (decayDb / 20.f)));
    const float pole = std::exp(lambda / (timeMs / 1000.f) / samplerate);
    SetPole(pole);
}

void OnePoleFilter::SetLowpass(float cutoff)
{
    assert(cutoff >= 0.f && cutoff <= 1.f);
    const float wc = TWO_PI * cutoff;
    const float y = 1 - std::cos(wc);
    const float p = -y + std::sqrt(y * y + 2 * y);
    SetPole(1 - p);
}

float OnePoleFilter::Tick(float in)
{
    state_[0] = gain_ * in * b0_ - state_[1] * a1_;
    state_[1] = state_[0];
    return state_[0];
}

void OnePoleFilter::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // OnePoleFilter only supports single channel input/output

    auto input_span = input.GetChannelSpan(0);
    auto output_span = output.GetChannelSpan(0);
    for (auto i = 0; i < input_span.size(); ++i)
    {
        output_span[i] = Tick(input_span[i]);
    }
}
uint32_t OnePoleFilter::InputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel input
}

uint32_t OnePoleFilter::OutputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel output
}

} // namespace sfFDN