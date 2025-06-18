#pragma once

#include <cstddef>
#include <span>

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delay.h"

namespace sfFDN
{

class SchroederAllpass
{
  public:
    SchroederAllpass(size_t delay, float g);

    void SetDelay(size_t delay);
    void SetG(float g);

    float Tick(float input);
    void ProcessBlock(std::span<const float> in, std::span<float> out);

  private:
    Delay delay_;
    float g_;
};

class SchroederAllpassSection : public AudioProcessor
{
  public:
    SchroederAllpassSection(size_t N);

    void SetDelays(std::span<size_t> delays);
    void SetGains(std::span<float> gains);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    size_t InputChannelCount() const override;

    size_t OutputChannelCount() const override;

  private:
    std::vector<SchroederAllpass> allpasses_;
    size_t stage_;
};
} // namespace sfFDN