#pragma once

#include "sffdn/audio_buffer.h"

namespace sfFDN
{
class Generator
{
  public:
    Generator() = default;
    virtual ~Generator() = default;

    virtual void Generate(AudioBuffer& output) = 0;
};

class SineWave : public Generator
{
  public:
    SineWave(float frequency, uint32_t sample_rate, float initial_phase = 0.0f);

    void SetFrequency(float frequency);

    void Generate(AudioBuffer& output) override;

  private:
    uint32_t sample_rate_;
    float phase_;
    float phase_increment_;
};

} // namespace sfFDN