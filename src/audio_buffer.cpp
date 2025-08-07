#include "sffdn/audio_buffer.h"

#include <cassert>
#include <cstddef>
#include <span>

namespace sfFDN
{

AudioBuffer::AudioBuffer()
    : size_(0)
    , channel_count_(0)
    , buffer_(nullptr)
    , buffer_span_{buffer_, 0}
{
}

AudioBuffer::AudioBuffer(std::span<float> buffer)
    : size_(buffer.size())
    , channel_count_(1)
    , buffer_(buffer.data())
    , buffer_span_{buffer.data(), buffer.size()}
{
}

AudioBuffer::AudioBuffer(uint32_t size, uint32_t channels, float* const buffer)
    : size_(size)
    , channel_count_(channels)
    , buffer_(buffer)
    , buffer_span_{buffer, size * channels}
{
}

AudioBuffer::AudioBuffer(uint32_t frame_size, uint32_t channels, std::span<float> buffer)
    : size_(frame_size)
    , channel_count_(channels)
    , buffer_(buffer.data())
    , buffer_span_{buffer.data(), buffer.size()}
{
    assert(buffer.size() >= frame_size * channels);
}

uint32_t AudioBuffer::SampleCount() const
{
    return size_;
}

uint32_t AudioBuffer::ChannelCount() const
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

std::span<const float> AudioBuffer::GetChannelSpan(uint32_t channel) const
{
    assert(channel < channel_count_);
    return buffer_span_.subspan(channel * size_, size_);
}

std::span<float> AudioBuffer::GetChannelSpan(uint32_t channel)
{
    assert(channel < channel_count_);
    return buffer_span_.subspan(channel * size_, size_);
}

AudioBuffer AudioBuffer::GetChannelBuffer(uint32_t channel) const
{
    assert(channel < channel_count_);
    return {size_, 1, buffer_ + channel * size_};
}
} // namespace sfFDN