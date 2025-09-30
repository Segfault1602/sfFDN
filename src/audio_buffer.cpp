#include "sffdn/audio_buffer.h"

#include <cassert>
#include <cstdint>
#include <span>

namespace sfFDN
{

AudioBuffer::AudioBuffer()
    : frame_size_(0)
    , channel_count_(0)
    , offset_(0)
    , chunk_size_(0)
{
}

AudioBuffer::AudioBuffer(std::span<float> buffer)
    : frame_size_(buffer.size())
    , channel_count_(1)
    , buffer_(buffer)
    , offset_(0)
    , chunk_size_(frame_size_)
{
    assert(buffer.data() != nullptr);
    assert(!buffer.empty());
}

AudioBuffer::AudioBuffer(uint32_t frame_size, uint32_t channels, std::span<float> buffer)
    : frame_size_(frame_size)
    , channel_count_(channels)
    , buffer_(buffer)
    , offset_(0)
    , chunk_size_(frame_size)
{
    assert(buffer.size() >= frame_size * channels);
}

uint32_t AudioBuffer::SampleCount() const
{
    return chunk_size_;
}

uint32_t AudioBuffer::ChannelCount() const
{
    return channel_count_;
}

float* AudioBuffer::Data()
{
    return buffer_.data();
}

const float* AudioBuffer::Data() const
{
    return buffer_.data();
}

std::span<const float> AudioBuffer::GetChannelSpan(uint32_t channel) const
{
    assert(channel < channel_count_);
    auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return channel_span;
}

std::span<float> AudioBuffer::GetChannelSpan(uint32_t channel)
{
    assert(channel < channel_count_);
    auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return channel_span;
}

AudioBuffer AudioBuffer::GetChannelBuffer(uint32_t channel) const
{
    assert(channel < channel_count_);
    auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return AudioBuffer(channel_span);
}

AudioBuffer AudioBuffer::Offset(uint32_t offset, uint32_t size) const
{
    AudioBuffer offset_buffer = *this;

    offset_buffer.offset_ = offset_ + offset;
    offset_buffer.chunk_size_ = size;
    return offset_buffer;
}

} // namespace sfFDN