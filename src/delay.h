#pragma once

#include <span>
#include <vector>

#include "audio_buffer.h"
#include "audio_processor.h"

namespace sfFDN
{

class Delay
{
  public:
    Delay(size_t delay = 0, size_t maxDelay = 4095);

    ~Delay();

    void Clear();

    void SetMaximumDelay(unsigned long delay);

    void SetDelay(size_t delay);

    unsigned long GetDelay(void) const
    {
        return delay_;
    };

    float LastOut(void) const
    {
        return lastFrame_;
    };

    float NextOut() const;

    float Tick(float input);

    void Process(const AudioBuffer input, AudioBuffer& output);

    void AddNextInputs(std::span<const float> input);
    void GetNextOutputs(std::span<float> output);

  protected:
    size_t inPoint_;
    size_t outPoint_;
    size_t delay_;
    std::vector<float> buffer_;
    float lastFrame_;
};

} // namespace sfFDN