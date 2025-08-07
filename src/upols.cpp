#include "upols.h"

#include <cassert>
#include <cmath>
#include <iostream>

#include "array_math.h"
#include "math_utils.h"

namespace sfFDN
{

UPOLS::UPOLS(uint32_t block_size, std::span<const float> fir)
    : block_size_(block_size)
    , fft_size_(Math::NextPowerOfTwo(block_size * 2))
    , fft_(fft_size_)
    , inputs_z_index_(0)
    , samples_needed_(block_size_)
{

    work_buffer_ = fft_.AllocateRealBuffer();
    result_buffer_ = fft_.AllocateRealBuffer();
    spectrum_buffer_ = fft_.AllocateComplexBuffer();
    assert(work_buffer_.data() != nullptr);
    assert(result_buffer_.data() != nullptr);
    assert(spectrum_buffer_.data() != nullptr);

    // Filter partition
    uint32_t filter_size = fir.size();

    uint32_t filter_count = std::ceil(static_cast<float>(filter_size) / block_size);

    for (auto i = 0; i < filter_count; ++i)
    {
        uint32_t filter_block_size = std::min(block_size, filter_size - i * block_size);

        auto fir_span = fir.subspan(i * block_size, filter_block_size);
        std::fill(work_buffer_.begin(), work_buffer_.end(), 0.f);
        std::copy(fir_span.begin(), fir_span.end(), work_buffer_.begin());

        auto filter_z = fft_.AllocateComplexBuffer();
        fft_.Forward(work_buffer_, filter_z);

        assert(filter_z.data() != nullptr);
        filters_z_.push_back(filter_z);

        auto input_z = fft_.AllocateComplexBuffer();
        assert(input_z.data() != nullptr);
        std::fill(input_z.begin(), input_z.end(), 0.f);
        inputs_z_.push_back(input_z);
    }

    std::fill(work_buffer_.begin(), work_buffer_.end(), 0.f);
}

UPOLS::~UPOLS()
{
    fft_.FreeBuffer(work_buffer_.data());
    fft_.FreeBuffer(result_buffer_.data());
    fft_.FreeBuffer(spectrum_buffer_.data());

    for (auto& filter_z : filters_z_)
    {
        fft_.FreeBuffer(filter_z.data());
    }
    for (auto& input_z : inputs_z_)
    {
        fft_.FreeBuffer(input_z.data());
    }
}

UPOLS::UPOLS(UPOLS&& other)
    : block_size_(other.block_size_)
    , fft_size_(other.fft_size_)
    , fft_(other.fft_size_)
    , inputs_z_index_(other.inputs_z_index_)
    , samples_needed_(other.samples_needed_)
{
    work_buffer_ = other.work_buffer_;
    other.work_buffer_ = std::span<float>{};

    result_buffer_ = other.result_buffer_;
    other.result_buffer_ = std::span<float>{};

    spectrum_buffer_ = other.spectrum_buffer_;
    other.spectrum_buffer_ = std::span<complex_t>{};

    filters_z_ = std::move(other.filters_z_);
    inputs_z_ = std::move(other.inputs_z_);
}

void UPOLS::Process(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == block_size_);
    assert(work_buffer_.size() == 2 * block_size_);

    // Prepare the input buffer for FFT
    std::copy(work_buffer_.begin() + block_size_, work_buffer_.end(), work_buffer_.begin());
    std::copy(input.begin(), input.end(), work_buffer_.end() - block_size_);

    if (inputs_z_index_ == 0)
    {
        inputs_z_index_ = inputs_z_.size() - 1;
    }
    else
    {
        --inputs_z_index_;
    }

    std::span<complex_t> input_z = inputs_z_[inputs_z_index_];
    fft_.Forward(work_buffer_, input_z);

    // Convolve with the filters
    std::fill(spectrum_buffer_.begin(), spectrum_buffer_.end(), 0.f);
    for (auto i = 0; i < filters_z_.size(); ++i)
    {
        auto& filter_z = filters_z_[i];
        auto& input_z = inputs_z_[(inputs_z_index_ + i) % inputs_z_.size()];

        fft_.ConvolveAccumulate(input_z, filter_z, spectrum_buffer_);
    }

    // Inverse FFT
    fft_.Inverse(spectrum_buffer_, result_buffer_);
    std::copy(result_buffer_.end() - block_size_, result_buffer_.end(), output.begin());
}

std::span<float> UPOLS::PrepareWorkBuffer()
{
    // Prepare the input buffer for FFT
    std::copy(work_buffer_.begin() + block_size_, work_buffer_.end(), work_buffer_.begin());
    return work_buffer_.subspan(block_size_, block_size_);
}

void UPOLS::AddSamples(std::span<const float> input)
{
    assert(samples_needed_ >= input.size());

    std::copy(input.begin(), input.end(), work_buffer_.end() - samples_needed_);
    samples_needed_ -= input.size();
}

bool UPOLS::IsReady() const
{
    return samples_needed_ == 0;
}

void UPOLS::Process(std::span<float> output)
{
    assert(output.size() == block_size_);
    assert(work_buffer_.size() == 2 * block_size_);
    assert(samples_needed_ == 0);

    if (inputs_z_index_ == 0)
    {
        inputs_z_index_ = inputs_z_.size() - 1;
    }
    else
    {
        --inputs_z_index_;
    }

    std::span<complex_t> input_z = inputs_z_[inputs_z_index_];
    fft_.Forward(work_buffer_, input_z);

    // Convolve with the filters
    std::fill(spectrum_buffer_.begin(), spectrum_buffer_.end(), 0.f);
    for (auto i = 0; i < filters_z_.size(); ++i)
    {
        auto& filter_z = filters_z_[i];
        auto& input_z = inputs_z_[(inputs_z_index_ + i) % inputs_z_.size()];

        fft_.ConvolveAccumulate(input_z, filter_z, spectrum_buffer_);
    }

    // Inverse FFT
    fft_.Inverse(spectrum_buffer_, result_buffer_);
    std::copy(result_buffer_.end() - block_size_, result_buffer_.end(), output.begin());

    std::copy(work_buffer_.begin() + block_size_, work_buffer_.end(), work_buffer_.begin());
    samples_needed_ = block_size_;
}

void UPOLS::PrintPartition() const
{
    std::cout << "[(" << fft_size_ << ") ";
    for (auto i = 0; i < filters_z_.size(); ++i)
    {
        std::cout << block_size_;
        if (i < filters_z_.size() - 1)
        {
            std::cout << "|";
        }
    }
    std::cout << "]" << std::endl;
}
} // namespace sfFDN