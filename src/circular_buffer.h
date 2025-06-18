#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace fdn
{

class CircularBuffer
{
  public:
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
    void Write(std::span<const float> data, size_t offset = 0);

    /// @brief Accumulate the specified number of samples into the buffer.
    /// @param data
    void Accumulate(std::span<const float> data, size_t offset = 0);

    /// @brief Read the specified number of samples from the buffer.
    /// @param data the buffer to read into
    /// @param offset the offset from the write pointer to start reading
    void Read(std::span<float> data, size_t offset = 0);

    /// @brief The total size of the buffer.
    /// @return the size of the buffer
    size_t Size() const;

    void Print() const;

  private:
    std::vector<float> buffer_;
    size_t write_ptr_;
};
} // namespace fdn