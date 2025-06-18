#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "fft.h"

namespace sfFDN
{

class UPOLS
{
  public:
    UPOLS(size_t block_size, std::span<const float> fir);
    ~UPOLS();

    UPOLS(UPOLS&& other);

    void Process(std::span<const float> input, std::span<float> output);

    std::span<float> PrepareWorkBuffer();
    void Process(std::span<float> output);

    void PrintPartition() const;

  private:
    size_t block_size_;
    size_t fft_size_;
    FFT fft_;

    std::vector<std::span<complex_t>> filters_z_;
    std::vector<std::span<complex_t>> inputs_z_; // Frequency domain delay line
    size_t inputs_z_index_;

    std::span<float> work_buffer_;
    std::span<complex_t> spectrum_buffer_;
    std::span<float> result_buffer_;
};
} // namespace sfFDN