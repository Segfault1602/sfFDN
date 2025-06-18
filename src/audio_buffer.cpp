#include "audio_buffer.h"

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace fdn
{

AudioBuffer::AudioBuffer()
    : size_(0)
    , channel_count_(0)
    , buffer_(nullptr)
{
}

AudioBuffer::AudioBuffer(std::span<float> buffer)
    : size_(buffer.size())
    , channel_count_(1)
    , buffer_(buffer.data())
{
}

AudioBuffer::AudioBuffer(std::span<const float> buffer)
    : size_(buffer.size())
    , channel_count_(1)
    , buffer_(const_cast<float*>(buffer.data()))
{
}

AudioBuffer::AudioBuffer(size_t size, size_t channels, float* buffer)
    : size_(size)
    , channel_count_(channels)
    , buffer_(buffer)
{
}

AudioBuffer::AudioBuffer(size_t size, size_t channels, const float* buffer)
    : size_(size)
    , channel_count_(channels)
    , buffer_(const_cast<float*>(buffer))
{
}

AudioBuffer::AudioBuffer(size_t frame_size, size_t channels, std::span<float> buffer)
    : size_(frame_size)
    , channel_count_(channels)
    , buffer_(buffer.data())
{
    assert(buffer.size() >= frame_size * channels);
}

size_t AudioBuffer::SampleCount() const
{
    return size_;
}

size_t AudioBuffer::ChannelCount() const
{
    return channel_count_;
}

float* AudioBuffer::Data()
{
    return buffer_;
}

const float* AudioBuffer::Data() const
{
    return buffer_;
}

std::span<const float> AudioBuffer::GetChannelSpan(size_t channel) const
{
    assert(channel < channel_count_);
    return std::span<const float>(buffer_ + channel * size_, size_);
}

std::span<float> AudioBuffer::GetChannelSpan(size_t channel)
{
    assert(channel < channel_count_);
    return std::span<float>(buffer_ + channel * size_, size_);
}

AudioBuffer AudioBuffer::GetChannelBuffer(size_t channel) const
{
    assert(channel < channel_count_);
    return AudioBuffer(size_, 1, buffer_ + channel * size_);
}

AudioBuffer AudioBuffer::GetChannelBuffer(size_t channel)
{
    assert(channel < channel_count_);
    return AudioBuffer(size_, 1, buffer_ + channel * size_);
}
} // namespace fdn