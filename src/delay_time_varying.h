#pragma once

#include <span>

#include "sffdn/delaya.h"

namespace sfFDN
{
class DelayTimeVarying
{
  public:
    DelayTimeVarying(float delay = 0.5, uint32_t maxDelay = 4095, uint32_t samplerate = 48000);
    ~DelayTimeVarying() = default;

    void Clear(void);

    void SetMaximumDelay(uint32_t delay);
    void SetDelay(float delay);

    void SetMod(float freq, float depth);

    float Tick(float input);

    void AddNextInput(float input);
    float GetNextOutput();

  private:
    void UpdateDelay();

    DelayAllpass delayA_;
    float delay_;

    uint32_t samplerate_;

    float mod_freq_;
    float mod_depth_;
    float mod_phase_;
    float mod_phase_increment_;
};
} // namespace sfFDN