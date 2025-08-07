// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <span>
#include <cstdint>

namespace sfFDN
{
/// @brief A class representing an audio buffer with multiple channels of non-interleaved audio data.
/// The AudioBuffer class does not own the underlying data and expects the data to stay valid for the lifetime of the
/// AudioBuffer instance.
class AudioBuffer
{
  public:
    /// @brief Constructs an empty audio buffer.
    AudioBuffer();

    /// @brief Constructs a mono audio buffer.
    explicit AudioBuffer(std::span<float> buffer);

    /// @brief Constructs a multi-channel audio buffer.
    /// @param frame_size The number of frames (samples) per channel.
    /// @param channels The number of channels.
    /// @param buffer A pointer to the interleaved audio data. The buffer must be large enough to hold `frame_size *
    /// channels` samples.
    AudioBuffer(uint32_t frame_size, uint32_t channels, float* const buffer);

    /// @brief Constructs a multi-channel audio buffer.
    /// @param frame_size The number of frames (samples) per channel.
    /// @param channels The number of channels.
    /// @param buffer A span representing the interleaved audio data. The span must be large enough to hold `frame_size
    /// * channels` samples.
    AudioBuffer(uint32_t frame_size, uint32_t channels, std::span<float> buffer);

    /// @brief Returns the number of samples in one channel of the audio buffer.
    /// @return The number of samples in one channel.
    uint32_t SampleCount() const;

    /// @brief Returns the number of channels in the audio buffer.
    /// @return The number of channels.
    uint32_t ChannelCount() const;

    /// @brief Provides direct access to the audio data.
    /// @return A pointer to the audio data.
    float* Data();

    /// @brief Provides direct access to the audio data.
    /// @return A pointer to the audio data.
    const float* Data() const;

    /// @brief Returns a span representing the audio data for a specific channel.
    /// @param channel The index of the channel to access.
    /// @return A span containing the audio data for the specified channel.
    std::span<const float> GetChannelSpan(uint32_t channel) const;

    /// @brief Returns a span representing the audio data for a specific channel.
    /// @param channel The index of the channel to access.
    /// @return A span containing the audio data for the specified channel.
    std::span<float> GetChannelSpan(uint32_t channel);

    /// @brief Returns an AudioBuffer object representing the audio data for a specific channel.
    /// @param channel The index of the channel to access.
    /// @return An AudioBuffer object containing the audio data for the specified channel.
    AudioBuffer GetChannelBuffer(uint32_t channel) const;

  private:
    uint32_t size_;
    uint32_t channel_count_;
    float* buffer_;
    std::span<float> buffer_span_;
};
} // namespace sfFDN