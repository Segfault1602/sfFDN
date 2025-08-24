// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>

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

    /// @brief Returns a new AudioBuffer where every channel is offset by a certain number of samples
    AudioBuffer Offset(uint32_t offset, uint32_t frame_size) const;

    class Iterator
    {
      public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::span<float>;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        Iterator(AudioBuffer* parent, uint32_t channel_index = 0);

        std::span<float> operator*() const;

        Iterator& operator++();

        Iterator operator++(int);

        bool operator==(const Iterator& other) const;

      private:
        AudioBuffer* parent_ = nullptr;
        uint32_t channel_index_ = 0;
    };

    class ConstIterator
    {
      public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::span<const float>;
        using difference_type = std::ptrdiff_t;

        ConstIterator() = default;

        ConstIterator(const AudioBuffer* parent, uint32_t channel_index = 0);

        std::span<const float> operator*() const;

        ConstIterator& operator++();

        ConstIterator operator++(int);

        bool operator==(const ConstIterator& other) const;

      private:
        const AudioBuffer* parent_ = nullptr;
        uint32_t channel_index_ = 0;
    };

    ConstIterator begin() const
    {
        return ConstIterator(this, 0);
    }

    ConstIterator end() const
    {
        return ConstIterator(this, channel_count_);
    }

    Iterator begin()
    {
        return Iterator(this, 0);
    }

    Iterator end()
    {
        return Iterator(this, channel_count_);
    }

    // Range support for C++20 std::views::zip
    auto size() const
    {
        return channel_count_;
    }

  private:
    uint32_t frame_size_;
    uint32_t channel_count_;
    std::span<float> buffer_;

    uint32_t offset_;
    uint32_t chunk_size_;
};
} // namespace sfFDN