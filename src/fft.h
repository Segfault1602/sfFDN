#pragma once

#include "sffdn/attributes.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

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
    FFT();
    ~FFT();

    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;

    FFT(FFT&& other) noexcept;
    FFT& operator=(FFT&& other) noexcept;

    bool Initialize(uint32_t fft_size);

    void Forward(const FFTRealBuffer& input, FFTComplexBuffer& spectrum) noexcept SFFDN_NONBLOCKING;
    void Inverse(const FFTComplexBuffer& spectrum, FFTRealBuffer& output) noexcept SFFDN_NONBLOCKING;

    void ConvolveAccumulate(const FFTComplexBuffer& dft_a, const FFTComplexBuffer& dft_b,
                            FFTComplexBuffer& dft_ab) noexcept SFFDN_NONBLOCKING;

    [[nodiscard]] FFTRealBuffer AllocateRealBuffer() const;
    [[nodiscard]] FFTComplexBuffer AllocateComplexBuffer() const;

  private:
    PFFFT_Setup* setup_{nullptr};
    uint32_t fft_size_{0};
    uint32_t complex_sample_count_{0};
    float* work_buffer_{nullptr};

    void Cleanup();
};

template <typename T>
class FFTBuffer
{
  public:
    FFTBuffer();
    FFTBuffer(std::span<T> buffer);
    ~FFTBuffer();

    std::span<T> Data() noexcept SFFDN_NONBLOCKING;
    std::span<const T> Data() const noexcept SFFDN_NONBLOCKING;

    FFTBuffer(const FFTBuffer&) = delete;
    FFTBuffer& operator=(const FFTBuffer&) = delete;
    FFTBuffer(FFTBuffer&& other) noexcept;
    FFTBuffer& operator=(FFTBuffer&& other) noexcept;

    std::span<T>::iterator begin() noexcept SFFDN_NONBLOCKING
    {
        return buffer_.begin();
    }
    std::span<T>::iterator end() noexcept SFFDN_NONBLOCKING
    {
        return buffer_.end();
    }

    constexpr std::span<T>::iterator begin() const noexcept SFFDN_NONBLOCKING
    {
        return buffer_.begin();
    }

    constexpr std::span<T>::iterator end() const noexcept SFFDN_NONBLOCKING
    {
        return buffer_.end();
    }

    uint32_t size() const noexcept SFFDN_NONBLOCKING
    {
        return buffer_.size();
    }

  private:
    std::span<T> buffer_;
};

} // namespace sfFDN
