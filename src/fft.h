#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

// #include <pffft.h>

struct PFFFT_Setup;

using complex_t = std::complex<float>;

namespace sfFDN
{

template <typename T>
class FFTBuffer;

using FFTRealBuffer = FFTBuffer<float>;
using FFTComplexBuffer = FFTBuffer<complex_t>;

class FFT
{
  public:
    FFT(uint32_t fft_size);
    ~FFT();

    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;
    FFT(FFT&&) = delete;
    FFT& operator=(FFT&&) = delete;

    void Forward(const FFTRealBuffer& input, FFTComplexBuffer& spectrum);
    void Inverse(const FFTComplexBuffer& spectrum, FFTRealBuffer& output);

    void ConvolveAccumulate(const FFTComplexBuffer& dft_a, const FFTComplexBuffer& dft_b, FFTComplexBuffer& dft_ab);

    [[nodiscard]] FFTRealBuffer AllocateRealBuffer() const;
    [[nodiscard]] FFTComplexBuffer AllocateComplexBuffer() const;

  private:
    PFFFT_Setup* setup_;
    uint32_t fft_size_;
    uint32_t complex_sample_count_;
    float* work_buffer_;
};

template <typename T>
class FFTBuffer
{
  public:
    FFTBuffer();
    FFTBuffer(std::span<T> buffer);
    ~FFTBuffer();

    std::span<T> Data();
    std::span<const T> Data() const;

    FFTBuffer(const FFTBuffer&) = delete;
    FFTBuffer& operator=(const FFTBuffer&) = delete;
    FFTBuffer(FFTBuffer&&) noexcept;
    FFTBuffer& operator=(FFTBuffer&&) noexcept;

    std::span<T>::iterator begin()
    {
        return std::span<T>(buffer_, size_).begin();
    }
    std::span<T>::iterator end()
    {
        return std::span<T>(buffer_, size_).end();
    }

    constexpr std::span<T>::iterator begin() const
    {
        return std::span<T>(buffer_, size_).begin();
    }

    constexpr std::span<T>::iterator end() const
    {
        return std::span<T>(buffer_, size_).end();
    }

    uint32_t size() const
    {
        return size_;
    }

  private:
    T* buffer_;
    uint32_t size_;
};

} // namespace sfFDN
