#include "fft.h"

#include <pffft.h>

#include <algorithm>
#include <cassert>
#include <format>

namespace sfFDN
{

FFT::FFT(uint32_t fft_size)
    : fft_size_(fft_size)
    , complex_sample_count_(fft_size / 2)
{

    setup_ = pffft_new_setup(fft_size_, PFFFT_REAL);
    if (setup_ == nullptr)
    {
        throw std::runtime_error(std::format("FFT size ({}) is unsuitable for PFFFT", fft_size_));
    }

    work_buffer_ = static_cast<float*>(pffft_aligned_malloc(fft_size * sizeof(float)));
}

FFT::~FFT()
{
    if (setup_ != nullptr)
    {
        pffft_destroy_setup(setup_);
        setup_ = nullptr;
    }

    if (work_buffer_ != nullptr)
    {
        pffft_aligned_free(work_buffer_);
        work_buffer_ = nullptr;
    }
}

void FFT::Forward(std::span<float> input, std::span<complex_t> spectrum)
{
    assert(input.size() == fft_size_);
    assert(spectrum.size() == complex_sample_count_);

    pffft_transform(setup_, input.data(), reinterpret_cast<float*>(spectrum.data()), work_buffer_, PFFFT_FORWARD);
}

void FFT::Inverse(std::span<complex_t> spectrum, std::span<float> output)
{
    assert(spectrum.size() == complex_sample_count_);
    assert(output.size() == fft_size_);

    pffft_transform(setup_, reinterpret_cast<const float*>(spectrum.data()), output.data(), work_buffer_,
                    PFFFT_BACKWARD);

    // float scalar = 1.0f / fft_size_;
    // ArrayMath::Scale(output, scalar, output);
}

void FFT::ConvolveAccumulate(std::span<complex_t> dft_a, std::span<complex_t> dft_b, std::span<complex_t> dft_ab)
{
    assert(dft_a.size() == complex_sample_count_);
    assert(dft_b.size() == complex_sample_count_);
    assert(dft_ab.size() == complex_sample_count_);

    pffft_zconvolve_accumulate(setup_, reinterpret_cast<const float*>(dft_a.data()),
                               reinterpret_cast<const float*>(dft_b.data()), reinterpret_cast<float*>(dft_ab.data()),
                               1.0f / static_cast<float>(fft_size_));
}

void FFT::Reorder(std::span<complex_t> spectrum, std::span<complex_t> reordered_spectrum, bool forward)
{
    assert(spectrum.size() == complex_sample_count_);
    assert(reordered_spectrum.size() == complex_sample_count_);
    assert(spectrum.data() != reordered_spectrum.data());

    // Reorder the spectrum using the FFT's internal bit-reversal ordering
    pffft_zreorder(setup_, reinterpret_cast<const float*>(spectrum.data()),
                   reinterpret_cast<float*>(reordered_spectrum.data()), forward ? PFFFT_FORWARD : PFFFT_BACKWARD);
}

std::span<float> FFT::AllocateRealBuffer() const
{
    auto mem = std::span<float>(static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float))), fft_size_);
    std::ranges::fill(mem, 0.f);
    return mem;
}

std::span<complex_t> FFT::AllocateComplexBuffer() const
{
    auto mem =
        std::span<complex_t>(static_cast<complex_t*>(pffft_aligned_malloc(complex_sample_count_ * sizeof(complex_t))),
                             complex_sample_count_);
    std::ranges::fill(mem, complex_t{0.f, 0.f});
    return mem;
}

void FFT::FreeBuffer(void* buffer)
{
    if (buffer == nullptr)
    {
        return;
    }
    pffft_aligned_free(buffer);
}
} // namespace sfFDN