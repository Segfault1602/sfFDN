#include "upols.h"

#include "math_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

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

    // Filter partition
    const uint32_t filter_size = fir.size();

    const uint32_t filter_count = std::ceil(static_cast<float>(filter_size) / block_size);

    for (auto i = 0u; i < filter_count; ++i)
    {
        const uint32_t filter_block_size = std::min(block_size, filter_size - (i * block_size));

        auto fir_span = fir.subspan(i * block_size, filter_block_size);
        std::ranges::fill(work_buffer_.Data(), 0.f);
        std::ranges::copy(fir_span, work_buffer_.begin());

        auto filter_z = fft_.AllocateComplexBuffer();
        fft_.Forward(work_buffer_, filter_z);

        filters_z_.push_back(std::move(filter_z));

        auto input_z = fft_.AllocateComplexBuffer();
        std::ranges::fill(input_z.Data(), 0.f);
        inputs_z_.push_back(std::move(input_z));
    }

    std::ranges::fill(work_buffer_, 0.f);
}

void UPOLS::Process(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == block_size_);
    assert(work_buffer_.Data().size() == 2 * block_size_);

    // Prepare the input buffer for FFT
    std::copy(work_buffer_.begin() + block_size_, work_buffer_.end(), work_buffer_.begin());
    std::ranges::copy(input, work_buffer_.end() - block_size_);

    if (inputs_z_index_ == 0)
    {
        inputs_z_index_ = inputs_z_.size() - 1;
    }
    else
    {
        --inputs_z_index_;
    }

    auto& input_z = inputs_z_[inputs_z_index_];
    fft_.Forward(work_buffer_, input_z);

    // Convolve with the filters
    std::ranges::fill(spectrum_buffer_.Data(), 0.f);
    for (auto i = 0u; i < filters_z_.size(); ++i)
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
    return work_buffer_.Data().subspan(block_size_, block_size_);
}

void UPOLS::AddSamples(std::span<const float> input)
{
    assert(samples_needed_ >= input.size());

    std::ranges::copy(input, work_buffer_.end() - samples_needed_);
    samples_needed_ -= input.size();
}

bool UPOLS::IsReady() const
{
    return samples_needed_ == 0;
}

void UPOLS::Process(std::span<float> output)
{
    assert(output.size() == block_size_);
    assert(work_buffer_.Data().size() == 2 * block_size_);
    assert(samples_needed_ == 0);

    if (inputs_z_index_ == 0)
    {
        inputs_z_index_ = inputs_z_.size() - 1;
    }
    else
    {
        --inputs_z_index_;
    }

    auto& input_z = inputs_z_[inputs_z_index_];
    fft_.Forward(work_buffer_, input_z);

    // Convolve with the filters
    std::ranges::fill(spectrum_buffer_.Data(), 0.f);
    for (auto i = 0u; i < filters_z_.size(); ++i)
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

void UPOLS::Clear()
{
    std::ranges::fill(work_buffer_, 0.f);
    std::ranges::fill(result_buffer_, 0.f);
    std::ranges::fill(spectrum_buffer_, complex_t{0.f, 0.f});
    for (auto& input_z : inputs_z_)
    {
        std::ranges::fill(input_z.Data(), complex_t{0.f, 0.f});
    }
    samples_needed_ = block_size_;
    inputs_z_index_ = 0;
}

void UPOLS::PrintPartition() const
{
    std::cout << "[(" << fft_size_ << ") ";
    for (auto i = 0u; i < filters_z_.size(); ++i)
    {
        std::cout << block_size_;
        if (i < filters_z_.size() - 1)
        {
            std::cout << "|";
        }
    }
    std::cout << "]\n";
}
} // namespace sfFDN