#pragma once

#include "fft.h"

#include <cstddef>
#include <span>
#include <vector>

namespace sfFDN
{
/**
 * @brief Uniform Partitioned Overlap-Save Convolver
 *
 */
class UPOLS
{
  public:
    UPOLS(uint32_t block_size, std::span<const float> fir);

    void Process(std::span<const float> input, std::span<float> output);

    std::span<float> PrepareWorkBuffer();

    void AddSamples(std::span<const float> input);
    bool IsReady() const;

    void Process(std::span<float> output);

    void Clear();

    void PrintPartition() const;
    std::string GetShortInfo() const;

  private:
    uint32_t block_size_;
    uint32_t fft_size_;
    FFT fft_;

    std::vector<FFTComplexBuffer> filters_z_;
    std::vector<FFTComplexBuffer> inputs_z_; // Frequency domain delay line
    uint32_t inputs_z_index_;

    FFTRealBuffer work_buffer_;
    FFTComplexBuffer spectrum_buffer_;
    FFTRealBuffer result_buffer_;

    uint32_t samples_needed_;
};
} // namespace sfFDN