#include "sffdn/audio_buffer.h"

#include "audio_buffer_alias.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <span>

namespace sfFDN
{
namespace
{
AudioBufferAlias ClassifyNonEmptyChannelSpans(const AudioBuffer& first,
                                              const AudioBuffer& second) noexcept SFFDN_NONBLOCKING
{
    const float* first_begin = first.GetChannelSpan(0).data();
    const float* second_begin = second.GetChannelSpan(0).data();
    const bool exact = first.ChannelCount() == second.ChannelCount() && first.SampleCount() == second.SampleCount() &&
                       first_begin == second_begin &&
                       (first.ChannelCount() == 1 || first.ChannelStride() == second.ChannelStride());
    if (exact)
    {
        return AudioBufferAlias::Exact;
    }

    // Channel spans are equal-length and start at increasing addresses, so each buffer's samples lie within
    // [first sample of channel 0, end of last channel).
    const float* first_end = std::to_address(first.GetChannelSpan(first.ChannelCount() - 1).end());
    const float* second_end = std::to_address(second.GetChannelSpan(second.ChannelCount() - 1).end());
    if (!std::less{}(first_begin, second_end) || !std::less{}(second_begin, first_end))
    {
        return AudioBufferAlias::Disjoint;
    }

    return AudioBufferAlias::Invalid;
}
} // namespace

AudioBuffer::AudioBuffer() noexcept SFFDN_NONBLOCKING : frame_size_(0), channel_count_(0), offset_(0), chunk_size_(0)
{
}

AudioBuffer::AudioBuffer(std::span<float> buffer) noexcept SFFDN_NONBLOCKING : frame_size_(buffer.size()),
                                                                               channel_count_(1),
                                                                               buffer_(buffer),
                                                                               offset_(0),
                                                                               chunk_size_(frame_size_)
{
    assert(buffer.data() != nullptr);
    assert(!buffer.empty());
}

AudioBuffer::AudioBuffer(uint32_t frame_size, uint32_t channels, std::span<float> buffer) noexcept SFFDN_NONBLOCKING
    : frame_size_(frame_size),
      channel_count_(channels),
      buffer_(buffer),
      offset_(0),
      chunk_size_(frame_size)
{
    assert(buffer.size() >= frame_size * channels);
}

uint32_t AudioBuffer::SampleCount() const noexcept SFFDN_NONBLOCKING
{
    return chunk_size_;
}

uint32_t AudioBuffer::ChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return channel_count_;
}

uint32_t AudioBuffer::ChannelStride() const noexcept SFFDN_NONBLOCKING
{
    return frame_size_;
}

bool AudioBuffer::IsPacked() const noexcept SFFDN_NONBLOCKING
{
    return channel_count_ <= 1 || ChannelStride() == SampleCount();
}

float* AudioBuffer::Data() noexcept SFFDN_NONBLOCKING
{
    return buffer_.subspan(offset_).data();
}

const float* AudioBuffer::Data() const noexcept SFFDN_NONBLOCKING
{
    return buffer_.subspan(offset_).data();
}

std::span<const float> AudioBuffer::GetChannelSpan(uint32_t channel) const noexcept SFFDN_NONBLOCKING
{
    assert(channel < channel_count_);
    const auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return channel_span;
}

std::span<float> AudioBuffer::GetChannelSpan(uint32_t channel) noexcept SFFDN_NONBLOCKING
{
    assert(channel < channel_count_);
    const auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return channel_span;
}

AudioBuffer AudioBuffer::GetChannelBuffer(uint32_t channel) const noexcept SFFDN_NONBLOCKING
{
    assert(channel < channel_count_);
    const auto channel_span = buffer_.subspan(channel * frame_size_, frame_size_).subspan(offset_, chunk_size_);
    return AudioBuffer(channel_span);
}

AudioBuffer AudioBuffer::Offset(uint32_t offset, uint32_t size) const noexcept SFFDN_NONBLOCKING
{
    AudioBuffer offset_buffer = *this;

    offset_buffer.offset_ = offset_ + offset;
    offset_buffer.chunk_size_ = size;
    return offset_buffer;
}

AudioBufferAlias ClassifyAudioBufferAlias(const AudioBuffer& first,
                                          const AudioBuffer& second) noexcept SFFDN_NONBLOCKING
{
    if (first.SampleCount() == 0 || first.ChannelCount() == 0 || second.SampleCount() == 0 ||
        second.ChannelCount() == 0)
    {
        return AudioBufferAlias::Disjoint;
    }

    return ClassifyNonEmptyChannelSpans(first, second);
}

} // namespace sfFDN