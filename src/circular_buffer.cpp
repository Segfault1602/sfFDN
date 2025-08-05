#include "circular_buffer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace sfFDN
{
CircularBuffer::CircularBuffer(size_t size)
    : buffer_(size, 0.f)
    , write_ptr_(0)
{
}

CircularBuffer::~CircularBuffer()
{
}

void CircularBuffer::Advance(size_t count)
{
    write_ptr_ = (write_ptr_ + count) % buffer_.size();
}

void CircularBuffer::Clear(size_t count, size_t offset)
{
    size_t start = (write_ptr_ + offset) % buffer_.size();
    size_t end = (start + count) % buffer_.size();

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

void CircularBuffer::Write(std::span<const float> data)
{
    assert(data.size() <= buffer_.size());
    size_t start = write_ptr_;
    size_t end = (start + data.size()) % buffer_.size();

    if (start < end)
    {
        std::copy(data.begin(), data.end(), buffer_.begin() + start);
    }
    else
    {
        size_t first_part = buffer_.size() - start;
        std::copy(data.begin(), data.begin() + first_part, buffer_.begin() + start);
        std::copy(data.begin() + first_part, data.end(), buffer_.begin());
    }
}

void CircularBuffer::Accumulate(std::span<const float> data, size_t offset)
{
    assert(data.size() <= buffer_.size());
    size_t start = (write_ptr_ + offset) % buffer_.size();
    size_t end = (start + data.size()) % buffer_.size();

    if (start < end)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            buffer_[i + start] += data[i];
        }
    }
    else
    {
        size_t first_part = buffer_.size() - start;
        for (size_t i = 0; i < first_part; ++i)
        {
            buffer_[i + start] += data[i];
        }

        for (size_t i = first_part; i < data.size(); ++i)
        {
            buffer_[i - first_part] += data[i];
        }
    }
}

void CircularBuffer::Read(std::span<float> data, bool clear_after_read)
{
    assert(data.size() <= buffer_.size());

    size_t start = (write_ptr_ + buffer_.size() - data.size()) % buffer_.size();
    size_t end = (start + data.size()) % buffer_.size();

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
        size_t first_part = buffer_.size() - start;
        std::copy(buffer_.begin() + start, buffer_.end(), data.begin());
        std::copy(buffer_.begin(), buffer_.begin() + end, data.begin() + first_part);

        if (clear_after_read)
        {
            std::fill(buffer_.begin() + start, buffer_.end(), 0.f);
            std::fill(buffer_.begin(), buffer_.begin() + end, 0.f);
        }
    }
}

size_t CircularBuffer::Size() const
{
    return buffer_.size();
}

} // namespace sfFDN