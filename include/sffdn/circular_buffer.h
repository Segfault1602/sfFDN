// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace sfFDN
{

/// @brief Circular buffer used to store audio samples.
/// The CircularBuffer class is not thread-safe and should not be used in a multi-threaded context.
class CircularBuffer
{
  public:
    /// @brief Constructs a CircularBuffer with the specified size.
    /// @param size the size of the buffer in samples
    CircularBuffer(size_t size = 32);
    ~CircularBuffer();

    /// @brief Advance the write pointer by the specified number of samples.
    /// @param count the number of samples to advance
    void Advance(size_t count);

    /// @brief Clear (set to zero) the specified number of samples in the buffer.
    /// @param count the number of samples to clear
    /// @param offset the offset from the write pointer to start clearing
    void Clear(size_t count, size_t offset = 0);

    /// @brief Write the specified number of samples to the buffer.
    /// @param data the data to write
    /// @note If the buffer is full, the oldest samples will be overwritten.
    void Write(std::span<const float> data);

    /// @brief Accumulate the specified number of samples into the buffer.
    /// @param data the data to accumulate
    /// @param offset the offset from the write pointer to start accumulating
    /// @note buffer[i] += data[i]
    /// @note The offset is added to the current write pointer, so it can be used to accumulate at a specific "future"
    /// position in the buffer.
    void Accumulate(std::span<const float> data, size_t offset = 0);

    /// @brief Read the specified number of samples from the buffer.
    /// @param data the buffer to read into
    /// @param clear_after_read if true, the read samples will be cleared (set to zero) in the buffer
    /// @note The `data.size()` most recently written samples will be read.
    void Read(std::span<float> data, bool clear_after_read = false);

    /// @brief The total size, in samples, of the buffer.
    /// @return the size of the buffer
    size_t Size() const;

  private:
    std::vector<float> buffer_;
    size_t write_ptr_;
};
} // namespace sfFDN