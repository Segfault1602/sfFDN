#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <span>

#include <pffft.h>

using complex_t = std::complex<float>;

namespace sfFDN
{
class FFT
{
  public:
    FFT(size_t fft_size);
    ~FFT();

    size_t GetNearestTransformSize(size_t fft_size) const;

    void Forward(std::span<float> input, std::span<complex_t> spectrum);
    void Inverse(std::span<complex_t> spectrum, std::span<float> output);

    void ConvolveAccumulate(std::span<complex_t> dft_a, std::span<complex_t> dft_b, std::span<complex_t> dft_ab);

    void FreeBuffer(void* buffer);

    std::span<float> AllocateRealBuffer();
    std::span<complex_t> AllocateComplexBuffer();

  private:
    PFFFT_Setup* setup_;
    size_t fft_size_;
    size_t complex_sample_count_;
    float* work_buffer_;
};
} // namespace sfFDN
