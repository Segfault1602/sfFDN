#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "audio_buffer.h"

namespace sfFDN
{
class AudioProcessor
{
  public:
    AudioProcessor() = default;
    virtual ~AudioProcessor() = default;

    /// @brief Process audio samples.
    /// @param input The input audio samples.
    /// @param output The output audio samples.
    virtual void Process(const AudioBuffer& input, AudioBuffer& output) = 0;

    virtual float ProcessSample(float input)
    {
        return 0;
    }

    virtual uint32_t InputChannelCount() const = 0;
    virtual uint32_t OutputChannelCount() const = 0;
};

class AudioProcessorChain : public AudioProcessor
{
  public:
    AudioProcessorChain(size_t block_size);
    ~AudioProcessorChain() override = default;

    bool AddProcessor(std::unique_ptr<AudioProcessor> processor);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;
    uint32_t OutputChannelCount() const override;

  private:
    size_t block_size_ = 0;
    std::vector<AudioProcessor*> processors_;

    std::vector<float> work_buffer_a_;
    std::vector<float> work_buffer_b_;
    size_t max_work_buffer_size_ = 0;
};
} // namespace sfFDN