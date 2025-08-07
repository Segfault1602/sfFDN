#pragma once

#include <complex>
#include <cstddef>
#include <span>

// #include <pffft.h>

struct PFFFT_Setup;

using complex_t = std::complex<float>;

namespace sfFDN
{
class FFT
{
  public:
    FFT(uint32_t fft_size);
    ~FFT();

    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;
    FFT(FFT&&) = delete;
    FFT& operator=(FFT&&) = delete;

    void Forward(std::span<float> input, std::span<complex_t> spectrum);
    void Inverse(std::span<complex_t> spectrum, std::span<float> output);

    void ConvolveAccumulate(std::span<complex_t> dft_a, std::span<complex_t> dft_b, std::span<complex_t> dft_ab);

    void Reorder(std::span<complex_t> spectrum, std::span<complex_t> reordered_spectrum, bool forward);

    [[nodiscard]] std::span<float> AllocateRealBuffer() const;
    [[nodiscard]] std::span<complex_t> AllocateComplexBuffer() const;
    static void FreeBuffer(void* buffer);

  private:
    PFFFT_Setup* setup_;
    uint32_t fft_size_;
    uint32_t complex_sample_count_;
    float* work_buffer_;
};
} // namespace sfFDN
