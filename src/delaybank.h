#pragma once

#include <span>
#include <vector>

#include "delay.h"

#include "audio_buffer.h"
#include "audio_processor.h"

namespace fdn
{
class DelayBank : public AudioProcessor
{
  public:
    DelayBank(unsigned long delayCount, unsigned long maxDelay = 4096);
    DelayBank(std::span<const size_t> delays, size_t block_size);
    ~DelayBank() = default;

    void Clear();
    void SetDelays(const std::span<const size_t> delays);

    size_t InputChannelCount() const override;
    size_t OutputChannelCount() const override;

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void AddNextInputs(const AudioBuffer& input);
    void GetNextOutputs(AudioBuffer& output);

  private:
    std::vector<fdn::Delay> delays_;
};
} // namespace fdn