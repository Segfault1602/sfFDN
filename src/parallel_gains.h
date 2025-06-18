#pragma once

#include <memory>
#include <span>
#include <vector>

#include "audio_processor.h"

namespace fdn
{

enum class ParallelGainsMode
{
    Multiplexed,  // Process input as a single channel and output to multiple channels
    DeMultiplexed // Process each input channel separately and output to multiple channels
};

class ParallelGains : public AudioProcessor
{
  public:
    ParallelGains(ParallelGainsMode mode);
    ParallelGains(size_t N, ParallelGainsMode mode, float gain = 1.0f);
    ParallelGains(ParallelGainsMode mode, std::span<const float> gains);

    void SetMode(ParallelGainsMode mode);
    void SetGains(std::span<const float> gains);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    size_t InputChannelCount() const override;
    size_t OutputChannelCount() const override;

  private:
    void ProcessBlockMultiplexed(const AudioBuffer& input, AudioBuffer& output);
    void ProcessBlockDeMultiplexed(const AudioBuffer& input, AudioBuffer& output);

    std::vector<float> gains_;
    ParallelGainsMode mode_;
};

} // namespace fdn