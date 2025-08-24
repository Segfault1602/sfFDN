#include "oscillator.h"

#include "pch.h"

namespace
{
constexpr uint32_t kSineTableSize = 1024; // Size of the sine table
std::array<float, kSineTableSize + 1> kSineTable;

struct SineTableInitializer
{
    SineTableInitializer()
    {
        for (auto i = 0; i < kSineTableSize; ++i)
        {
            kSineTable[i] = std::sinf((2.0f * std::numbers::pi * i) / kSineTableSize);
        }

        kSineTable[kSineTableSize] = 0.f;
    }
};
SineTableInitializer sine_table_initializer;

float Sine(float phase)
{
    assert(phase >= 0.0f && phase < 1.0f);
    float index = phase * kSineTableSize;

    int32_t uindex = static_cast<int32_t>(index);
    float frac = index - static_cast<float>(uindex);

    float a = kSineTable[uindex];
    float b = kSineTable[uindex + 1];
    return a + (b - a) * frac;
}

} // namespace

namespace sfFDN
{
SineWave::SineWave(float frequency, uint32_t sample_rate, float initial_phase)
    : frequency_(frequency)
    , sample_rate_(sample_rate)
    , phase_(initial_phase)
{
    // Normalize the initial phase to the range [0, 2 * PI)
    phase_increment_ = frequency_ / sample_rate_;
}

void SineWave::Generate(AudioBuffer& output)
{
    for (auto i = 0; i < output.SampleCount(); ++i)
    {
        float sample = Sine(phase_);

        for (auto channel = 0; channel < output.ChannelCount(); ++channel)
        {
            output.GetChannelSpan(channel)[i] = sample;
        }

        phase_ += phase_increment_;
        while (phase_ >= 1.0)
        {
            phase_ -= 1.0;
        }
    }
}
} // namespace sfFDN