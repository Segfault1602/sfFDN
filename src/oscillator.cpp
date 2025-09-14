#include "sffdn/oscillator.h"

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
    return a + ((b - a) * frac);
}

} // namespace

namespace sfFDN
{

SineWave::SineWave(float frequency, float initial_phase)
    : phase_(initial_phase)
    , phase_increment_(frequency)
    , amplitude_(1.0f)
    , offset_(0.0f)
    , phase_offset_(0.0f)
{
}

void SineWave::ResetPhase()
{
    phase_ = 0.0f;
}

void SineWave::SetFrequency(float frequency)
{
    phase_increment_ = frequency;
}

void SineWave::SetAmplitude(float amplitude)
{
    amplitude_ = amplitude;
}

void SineWave::SetOffset(float offset)
{
    offset_ = offset;
}

float SineWave::GetAmplitude() const
{
    return amplitude_;
}

float SineWave::GetOffset() const
{
    return offset_;
}

float SineWave::NextOut() const
{
    return (Sine(phase_ + phase_offset_) * amplitude_) + offset_;
}

float SineWave::Tick()
{
    float out = (Sine(phase_ + phase_offset_) * amplitude_) + offset_;
    phase_ += phase_increment_;
    phase_ -= std::floor(phase_);
    return out;
}

void SineWave::Generate(std::span<float> output)
{
    float phase = phase_;
    for (float& i : output)
    {
        i = (Sine(phase + phase_offset_) * amplitude_) + offset_;
        phase += phase_increment_;
    }
    phase_ = phase;
    phase_ -= std::floor(phase_);
}

void SineWave::Multiply(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());

    float phase = phase_;
    for (auto [in, out] : std::views::zip(input, output))
    {
        out = in * ((Sine(phase + phase_offset_) * amplitude_) + offset_);
        phase += phase_increment_;
    }
    phase_ = phase;
    phase_ -= std::floor(phase_);
}

void SineWave::MultiplyAccumulate(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());

    float phase = phase_;
    for (auto [in, out] : std::views::zip(input, output))
    {
        out += in * ((Sine(phase + phase_offset_) * amplitude_) + offset_);
        phase += phase_increment_;
    }
    phase_ = phase;
    phase_ -= std::floor(phase_);
}

} // namespace sfFDN