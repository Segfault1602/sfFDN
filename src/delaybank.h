#pragma once

#include <span>
#include <vector>

#include "delay.h"
#include "delay_time_varying.h"
#include "delaya.h"

namespace fdn
{
class DelayBank
{
  public:
    DelayBank(unsigned long delayCount, unsigned long maxDelay = 4096);
    DelayBank(const std::span<const float> delays, unsigned long maxDelay);
    ~DelayBank() = default;

    void Clear();
    void SetDelays(const std::span<const float> delays);
    void SetModulation(float freq, float depth);

    /**
     * @brief Tick the delay bank.
     * 'input' and 'output' must have the same size.
     * The size of 'input' must be a multiple of the delay count.
     * 'input' and 'output' can point to the same memory.
     *
     * @note Samples in the input buffer are interleaved by delay line:
     * input = [delay0_sample0, delay1_sample0, delay1_sample1, ...]
     *
     * @param input The input buffer.
     * @param output The output buffer.
     */
    void Tick(const std::span<const float> input, std::span<float> output);

    void AddNextInputs(const std::span<const float> input);
    void GetNextOutputs(std::span<float> output);

  private:
    std::vector<fdn::DelayTimeVarying> delays_;
};
} // namespace fdn