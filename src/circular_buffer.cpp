#include "circular_buffer.h"

#include "pch.h"

namespace sfFDN
{
CircularBuffer::CircularBuffer(uint32_t size)
    : buffer_(size, 0.f)
    , write_ptr_(0)
{
}

CircularBuffer::~CircularBuffer() = default;

void CircularBuffer::Advance(uint32_t count)
{
    write_ptr_ = (write_ptr_ + count) % buffer_.size();
}

void CircularBuffer::Clear(uint32_t count, uint32_t offset)
{
    const uint32_t start = (write_ptr_ + offset) % buffer_.size();
    const uint32_t end = (start + count) % buffer_.size();

    if (start < end)
    {
        std::fill(buffer_.begin() + start, buffer_.begin() + end, 0.f);
    }
    else
    {
        std::fill(buffer_.begin() + start, buffer_.end(), 0.f);
        std::fill(buffer_.begin(), buffer_.begin() + end, 0.f);
    }
}

void CircularBuffer::Clear()
{
    std::ranges::fill(buffer_, 0.f);
    write_ptr_ = 0;
}

void CircularBuffer::Write(std::span<const float> data)
{
    assert(data.size() <= buffer_.size());
    const uint32_t start = write_ptr_;
    const uint32_t end = (start + data.size()) % buffer_.size();

    if (start < end)
    {
        std::ranges::copy(data, buffer_.begin() + start);
    }
    else
    {
        const uint32_t first_part = buffer_.size() - start;
        std::copy(data.begin(), data.begin() + first_part, buffer_.begin() + start);
        std::copy(data.begin() + first_part, data.end(), buffer_.begin());
    }
}

void CircularBuffer::Accumulate(std::span<const float> data, uint32_t offset)
{
    assert(data.size() <= buffer_.size());
    const uint32_t start = (write_ptr_ + offset) % buffer_.size();
    const uint32_t end = (start + data.size()) % buffer_.size();

    if (start < end)
    {
        for (auto i = 0u; i < data.size(); ++i)
        {
            buffer_[i + start] += data[i];
        }
    }
    else
    {
        const uint32_t first_part = buffer_.size() - start;
        for (auto i = 0u; i < first_part; ++i)
        {
            buffer_[i + start] += data[i];
        }

        for (auto i = first_part; i < data.size(); ++i)
        {
            buffer_[i - first_part] += data[i];
        }
    }
}

void CircularBuffer::Read(std::span<float> data, bool clear_after_read)
{
    assert(data.size() <= buffer_.size());

    const uint32_t start = (write_ptr_ + buffer_.size() - data.size()) % buffer_.size();
    const uint32_t end = (start + data.size()) % buffer_.size();

    if (start < end)
    {
        std::copy(buffer_.begin() + start, buffer_.begin() + end, data.begin());
        if (clear_after_read)
        {
            std::fill(buffer_.begin() + start, buffer_.begin() + end, 0.f);
        }
    }
    else
    {
        const uint32_t first_part = buffer_.size() - start;
        std::copy(buffer_.begin() + start, buffer_.end(), data.begin());
        std::copy(buffer_.begin(), buffer_.begin() + end, data.begin() + first_part);

        if (clear_after_read)
        {
            std::fill(buffer_.begin() + start, buffer_.end(), 0.f);
            std::fill(buffer_.begin(), buffer_.begin() + end, 0.f);
        }
    }
}

uint32_t CircularBuffer::Size() const
{
    return buffer_.size();
}

} // namespace sfFDN