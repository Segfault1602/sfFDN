#pragma once

#include <span>

#include "delaya.h"

namespace fdn
{
class DelayTimeVarying
{
  public:
    DelayTimeVarying(float delay = 0.5, unsigned long maxDelay = 4095, size_t samplerate = 48000);
    ~DelayTimeVarying() = default;

    void Clear(void);

    void SetMaximumDelay(unsigned long delay);
    void SetDelay(float delay);

    void SetMod(float freq, float depth);

    float Tick(float input);

    void AddNextInput(float input);
    float GetNextOutput();

  private:
    void UpdateDelay();

    DelayAllpass delayA_;
    float delay_;

    size_t samplerate_;

    float mod_freq_;
    float mod_depth_;
    float mod_phase_;
    float mod_phase_increment_;
};
} // namespace fdn