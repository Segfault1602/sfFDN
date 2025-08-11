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
    , channel_buffers_{nullptr}
{
}

AudioBuffer::AudioBuffer(std::span<float> buffer)
    : size_(buffer.size())
    , channel_count_(1)
    , buffer_(buffer.data())
    , channel_buffers_{nullptr}
{
    assert(buffer.data() != nullptr);
    assert(buffer.size() > 0);
    channel_buffers_[0] = buffer.data();
}

AudioBuffer::AudioBuffer(uint32_t frame_size, uint32_t channels, float* buffer)
    : size_(frame_size)
    , channel_count_(channels)
    , buffer_(buffer)
    , channel_buffers_{nullptr}
{
    assert(buffer != nullptr);
    assert(frame_size > 0);
    assert(channels > 0);
    assert(channels <= channel_buffers_.size());

    auto buffer_span = std::span<float>(buffer_, frame_size * channels);

    for (uint32_t i = 0; i < channels; ++i)
    {
        auto channel_span = buffer_span.subspan(i * frame_size, frame_size);
        channel_buffers_.at(i) = channel_span.data();
    }
}

AudioBuffer::AudioBuffer(uint32_t frame_size, uint32_t channels, std::span<float> buffer)
    : size_(frame_size)
    , channel_count_(channels)
    , buffer_(buffer.data())
{
    assert(buffer.size() >= frame_size * channels);
    assert(channels <= channel_buffers_.size());

    for (uint32_t i = 0; i < channels; ++i)
    {
        channel_buffers_.at(i) = buffer.subspan(i * frame_size, frame_size).data();
    }
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
    return {channel_buffers_.at(channel), size_};
}

std::span<float> AudioBuffer::GetChannelSpan(uint32_t channel)
{
    assert(channel < channel_count_);
    return {channel_buffers_.at(channel), size_};
}

AudioBuffer AudioBuffer::GetChannelBuffer(uint32_t channel) const
{
    assert(channel < channel_count_);
    return {size_, 1, channel_buffers_.at(channel)};
}

AudioBuffer AudioBuffer::Offset(uint32_t offset, uint32_t size) const
{
    AudioBuffer offset_buffer = *this;

    offset_buffer.size_ = size;
    for (uint32_t i = 0; i < channel_count_; ++i)
    {
        offset_buffer.channel_buffers_.at(i) = channel_buffers_.at(i) + offset;
    }

    return offset_buffer;
}
} // namespace sfFDN