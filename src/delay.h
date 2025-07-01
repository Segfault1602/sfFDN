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
    Delay(uint32_t delay = 0, uint32_t maxDelay = 4095);

    ~Delay();

    void Clear();

    void SetMaximumDelay(unsigned long delay);

    void SetDelay(uint32_t delay);

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
    uint32_t inPoint_;
    uint32_t outPoint_;
    uint32_t delay_;
    std::vector<float> buffer_;
    float lastFrame_;
};

} // namespace sfFDN