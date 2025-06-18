#pragma once

#include <span>
#include <vector>

#include "audio_buffer.h"

namespace fdn
{
class DelayAllpass
{
  public:
    DelayAllpass(float delay = 0.5, size_t maxDelay = 4095);

    //! Class destructor.
    ~DelayAllpass();

    //! Clears all internal states of the delay line.
    void Clear(void);

    //! Get the maximum delay-line length.
    unsigned long GetMaximumDelay(void)
    {
        return buffer_.size() - 1;
    };

    void SetMaximumDelay(unsigned long delay);

    void SetDelay(float delay);

    float GetDelay(void) const
    {
        return delay_;
    };

    float LastOut(void) const
    {
        return lastFrame_;
    };

    float Tick(float input);

    void Process(const AudioBuffer& input, AudioBuffer& output);

  protected:
    unsigned long inPoint_;
    unsigned long outPoint_;
    float delay_;
    float alpha_;
    float coeff_;
    float apInput_;
    float nextOutput_;
    bool doNextOut_;

    std::vector<float> buffer_;
    float lastFrame_;
    float gain_;

  private:
    void UpdateAlpha(float delay);
    float NextOut();
};

} // namespace fdn