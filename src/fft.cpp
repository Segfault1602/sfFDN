#include "fft.h"

#include <pffft.h>

#include <algorithm>
#include <cassert>
#include <format>
#include <span>

namespace
{
float* AsFloatPtr(sfFDN::FFTComplexBuffer& buffer)
{
    return reinterpret_cast<float*>(buffer.Data().data());
}

const float* AsFloatPtr(const sfFDN::FFTComplexBuffer& buffer)
{
    return reinterpret_cast<const float*>(buffer.Data().data());
}
} // namespace

namespace sfFDN
{

template <typename T>
FFTBuffer<T>::FFTBuffer()
    : buffer_(nullptr)
    , size_(0)
{
}

template <typename T>
FFTBuffer<T>::FFTBuffer(std::span<T> buffer)
    : buffer_(buffer.data())
    , size_(buffer.size())
{
    assert(buffer.data() != nullptr);
    assert(buffer.size() > 0);
}

template <typename T>
FFTBuffer<T>::~FFTBuffer()
{
    if (buffer_ != nullptr)
    {
        pffft_aligned_free(static_cast<void*>(buffer_));
    }
    buffer_ = nullptr;
    size_ = 0;
}

template <typename T>
FFTBuffer<T>::FFTBuffer(FFTBuffer&& other) noexcept
    : buffer_(other.buffer_)
    , size_(other.size_)
{
    other.buffer_ = nullptr;
    other.size_ = 0;
}

template <typename T>
FFTBuffer<T>& FFTBuffer<T>::operator=(FFTBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (buffer_ != nullptr)
        {
            pffft_aligned_free(static_cast<void*>(buffer_));
        }
        buffer_ = other.buffer_;
        size_ = other.size_;
        other.buffer_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template <typename T>
std::span<T> FFTBuffer<T>::Data()
{
    assert(buffer_ != nullptr);
    return {buffer_, size_};
}

template <typename T>
std::span<const T> FFTBuffer<T>::Data() const
{
    assert(buffer_ != nullptr);
    return {buffer_, size_};
}

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

void FFT::Forward(const FFTRealBuffer& input, FFTComplexBuffer& spectrum)
{
    assert(input.Data().size() == fft_size_);
    assert(spectrum.Data().size() == complex_sample_count_);

    pffft_transform(setup_, input.Data().data(), AsFloatPtr(spectrum), work_buffer_, PFFFT_FORWARD);
}

void FFT::Inverse(const FFTComplexBuffer& spectrum, FFTRealBuffer& output)
{
    assert(spectrum.Data().size() == complex_sample_count_);
    assert(output.Data().size() == fft_size_);

    pffft_transform(setup_, AsFloatPtr(spectrum), output.Data().data(), work_buffer_, PFFFT_BACKWARD);
}

void FFT::ConvolveAccumulate(const FFTComplexBuffer& dft_a, const FFTComplexBuffer& dft_b, FFTComplexBuffer& dft_ab)
{
    assert(dft_a.Data().size() == complex_sample_count_);
    assert(dft_b.Data().size() == complex_sample_count_);
    assert(dft_ab.Data().size() == complex_sample_count_);

    pffft_zconvolve_accumulate(setup_, AsFloatPtr(dft_a), AsFloatPtr(dft_b), AsFloatPtr(dft_ab),
                               1.0f / static_cast<float>(fft_size_));
}

FFTRealBuffer FFT::AllocateRealBuffer() const
{
    auto mem = std::span<float>(static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float))), fft_size_);
    std::ranges::fill(mem, 0.f);
    return {mem};
}

FFTComplexBuffer FFT::AllocateComplexBuffer() const
{
    auto mem =
        std::span<complex_t>(static_cast<complex_t*>(pffft_aligned_malloc(complex_sample_count_ * sizeof(complex_t))),
                             complex_sample_count_);
    std::ranges::fill(mem, complex_t{0.f, 0.f});
    return {mem};
}

template class FFTBuffer<float>;
template class FFTBuffer<complex_t>;
} // namespace sfFDN