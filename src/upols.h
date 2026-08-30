#pragma once

#include "fft.h"
#include "sffdn/attributes.h"

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
    UPOLS();

    bool Initialize(uint32_t block_size, std::span<const float> fir);

    void Process(std::span<const float> input, std::span<float> output) noexcept SFFDN_NONBLOCKING;

    std::span<float> PrepareWorkBuffer() noexcept SFFDN_NONBLOCKING;

    void AddSamples(std::span<const float> input) noexcept SFFDN_NONBLOCKING;
    bool IsReady() const noexcept SFFDN_NONBLOCKING;

    void Process(std::span<float> output) noexcept SFFDN_NONBLOCKING;

    void Clear();

    void PrintPartition() const;
    std::string GetShortInfo() const;

  private:
    bool initialized_{false};
    uint32_t block_size_{0};
    uint32_t fft_size_{0};
    FFT fft_;

    std::vector<FFTComplexBuffer> filters_z_;
    std::vector<FFTComplexBuffer> inputs_z_; // Frequency domain delay line
    uint32_t inputs_z_index_{0};

    FFTRealBuffer work_buffer_;
    FFTComplexBuffer spectrum_buffer_;
    FFTRealBuffer result_buffer_;

    uint32_t samples_needed_{0};
};
} // namespace sfFDN