#pragma once

#include <cstddef>
#include <span>

namespace sfFDN
{
/// @brief A class representing an audio buffer with multiple channels of non-interleaved audio data.
class AudioBuffer
{
  public:
    AudioBuffer();
    AudioBuffer(std::span<float> buffer);
    AudioBuffer(std::span<const float> buffer);
    AudioBuffer(size_t frame_size, size_t channels, float* buffer);
    AudioBuffer(size_t frame_size, size_t channels, const float* buffer);
    AudioBuffer(size_t frame_size, size_t channels, std::span<float> buffer);

    /// @brief Returns the number of samples in one channel of the audio buffer.
    /// @return The number of samples in one channel.
    size_t SampleCount() const;

    /// @brief Returns the number of channels in the audio buffer.
    /// @return The number of channels.
    size_t ChannelCount() const;

    /// @brief Provides direct access to the audio data.
    float* Data();

    const float* Data() const;

    /// @brief Returns a span representing the audio data for a specific channel.
    /// @param channel The index of the channel to access.
    /// @return A span containing the audio data for the specified channel.
    std::span<const float> GetChannelSpan(size_t channel) const;

    /// @brief Returns a span representing the audio data for a specific channel.
    /// @param channel The index of the channel to access.
    /// @return A span containing the audio data for the specified channel.
    std::span<float> GetChannelSpan(size_t channel);

    AudioBuffer GetChannelBuffer(size_t channel) const;
    AudioBuffer GetChannelBuffer(size_t channel);

  private:
    const size_t size_;
    const size_t channel_count_;
    float* buffer_;
};
} // namespace sfFDN