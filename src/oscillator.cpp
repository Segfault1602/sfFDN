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
            kSineTable[i] = std::sin((2.0 * std::numbers::pi * i) / kSineTableSize);
        }

        kSineTable[kSineTableSize] = 0.f;
    }
};
SineTableInitializer sine_table_initializer;

float Sine(float phase)
{
    while (phase < 0.f)
    {
        phase += 1.f;
    }

    phase = phase - std::floor(phase);

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
    : sample_rate_(sample_rate)
    , phase_(initial_phase)
    , phase_increment_(frequency / sample_rate)
{
}

void SineWave::SetFrequency(float frequency)
{
    phase_increment_ = frequency / sample_rate_;
}

void SineWave::Generate(AudioBuffer& output)
{
    assert(output.ChannelCount() == 1);

    float phase = phase_;
    auto first_channel = output.GetChannelSpan(0);
    for (float& i : first_channel)
    {
        i = Sine(phase);
        phase += phase_increment_;
    }
    phase_ = phase;
    phase_ -= std::floor(phase_);
}
} // namespace sfFDN