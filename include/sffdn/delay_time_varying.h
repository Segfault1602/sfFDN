#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/delay_interp.h"
#include "sffdn/oscillator.h"

#include <span>
#include <variant>

namespace sfFDN
{
class DelayTimeVarying
{
  public:
    DelayTimeVarying(float delay = 0.5, uint32_t max_delay = 4095,
                     DelayInterpolationType type = DelayInterpolationType::Linear);

    void Clear();

    void SetMaximumDelay(uint32_t delay);
    void SetDelay(float delay);
    float GetDelay() const;

    void SetMod(float freq, float amplitude, float phase_offset = 0.0f);

    float Tick(float input);

    void Process(const AudioBuffer& input, AudioBuffer& output);

  private:
    void UpdateDelay();

    DelayInterp delay_;
    float base_delay_;

    SineWave lfo_;
};
} // namespace sfFDN