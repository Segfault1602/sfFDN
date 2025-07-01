#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "delay.h"

#include "audio_buffer.h"
#include "audio_processor.h"

namespace sfFDN
{
class DelayBank : public AudioProcessor
{
  public:
    DelayBank(unsigned long delayCount, unsigned long maxDelay = 4096);
    DelayBank(std::span<const uint32_t> delays, uint32_t block_size);
    ~DelayBank() = default;

    void Clear();
    void SetDelays(const std::span<const uint32_t> delays, uint32_t block_size = 512);

    uint32_t InputChannelCount() const override;
    uint32_t OutputChannelCount() const override;

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void AddNextInputs(const AudioBuffer& input);
    void GetNextOutputs(AudioBuffer& output);

  private:
    std::vector<sfFDN::Delay> delays_;
};
} // namespace sfFDN