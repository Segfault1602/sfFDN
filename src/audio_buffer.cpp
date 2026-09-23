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
    const bool exact = first.ChannelCount() == second.ChannelCount() && first.SampleCount() == second.SampleCount() &&
                       first.GetChannelSpan(0).data() == second.GetChannelSpan(0).data() &&
                       (first.ChannelCount() == 1 || first.ChannelStride() == second.ChannelStride());
    if (exact)
    {
        return AudioBufferAlias::Exact;
    }

    uint32_t first_channel = 0;
    uint32_t second_channel = 0;
    while (first_channel < first.ChannelCount() && second_channel < second.ChannelCount())
    {
        const auto first_span = first.GetChannelSpan(first_channel);
        const auto second_span = second.GetChannelSpan(second_channel);
        if (std::less{}(first_span.data(), std::to_address(second_span.end())) &&
            std::less{}(second_span.data(), std::to_address(first_span.end())))
        {
            return AudioBufferAlias::Partial;
        }

        if (std::less{}(std::to_address(first_span.end()), std::to_address(second_span.end())))
        {
            ++first_channel;
        }
        else
        {
            ++second_channel;
        }
    }

    return AudioBufferAlias::Disjoint;
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