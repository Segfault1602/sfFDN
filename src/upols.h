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
    UPOLS(uint32_t block_size, std::span<const float> fir);
    ~UPOLS();

    UPOLS(UPOLS&& other);

    void Process(std::span<const float> input, std::span<float> output);

    std::span<float> PrepareWorkBuffer();

    void AddSamples(std::span<const float> input);
    bool IsReady() const;

    void Process(std::span<float> output);

    void PrintPartition() const;

  private:
    uint32_t block_size_;
    uint32_t fft_size_;
    FFT fft_;

    std::vector<std::span<complex_t>> filters_z_;
    std::vector<std::span<complex_t>> inputs_z_; // Frequency domain delay line
    uint32_t inputs_z_index_;

    std::span<float> work_buffer_;
    std::span<complex_t> spectrum_buffer_;
    std::span<float> result_buffer_;

    uint32_t samples_needed_;
};
} // namespace sfFDN