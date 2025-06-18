#include "filter.h"

#include <cmath>
#include <iostream>
#include <numbers>

namespace
{
constexpr float TWO_PI = std::numbers::pi_v<float> * 2;
}

namespace sfFDN
{
void Filter::SetGain(float gain)
{
    gain_ = gain;
}

void Filter::SetA(const float (&a)[COEFFICIENT_COUNT])
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        a_[i] = a[i];
    }
}

void Filter::SetB(const float (&b)[COEFFICIENT_COUNT])
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        b_[i] = b[i];
    }
}

void Filter::Clear()
{
    for (size_t i = 0; i < COEFFICIENT_COUNT; ++i)
    {
        outputs_[i] = 0.f;
        inputs_[i] = 0.f;
    }
}

void Filter::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);

    for (size_t i = 0; i < size; ++i)
    {
        out[i] = Tick(in[i]);
    }
}

void OnePoleFilter::SetPole(float pole)
{
    // https://ccrma.stanford.edu/~jos/fp/One_Pole.html
    // If the filter has a pole at z = -a, then a_[1] will be -pole;
    assert(pole <= 1.f && pole >= -1.f);

    // Set the b value to 1 - |a| to get a peak gain of 1.
    b_[0] = 1.f - std::abs(pole);
    a_[1] = -pole;
}

void OnePoleFilter::SetCoefficients(float b0, float a1)
{
    b_[0] = b0;
    a_[1] = a1;
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
    outputs_[0] = gain_ * in * b_[0] - outputs_[1] * a_[1];
    outputs_[1] = outputs_[0];
    return outputs_[0];
}

void OnePoleFilter::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // OnePoleFilter only supports single channel input/output

    auto input_buf = input.GetChannelBuffer(0);
    auto output_buf = output.GetChannelBuffer(0);
    ProcessBlock(input_buf.Data(), output_buf.Data(), input.SampleCount());
}
size_t OnePoleFilter::InputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel input
}

size_t OnePoleFilter::OutputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel output
}

} // namespace sfFDN