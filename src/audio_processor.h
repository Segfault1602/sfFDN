#pragma once

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

    virtual uint32_t InputChannelCount() const = 0;
    virtual uint32_t OutputChannelCount() const = 0;
};
} // namespace sfFDN