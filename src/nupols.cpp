#include "nupols.h"

#include <cassert>
#include <iostream>

#include "array_math.h"

namespace sfFDN
{

NupolsSegment::NupolsSegment(size_t parent_block_size, size_t block_size, size_t delay, std::span<const float> fir)
    : upols_(block_size, fir)
    , delay_(delay)
    , block_size_(block_size)
    , sample_counter_(0)
{
    output_buffer_.resize(block_size, 0.f);

    size_t a = block_size / parent_block_size;
    assert(delay >= parent_block_size * (a - 1));
    deadline_offset_ = delay - parent_block_size * (a - 1);
}

NupolsSegment::NupolsSegment(NupolsSegment&& other)
    : upols_(std::move(other.upols_))
    , delay_(other.delay_)
{
    output_buffer_ = std::move(other.output_buffer_);
    sample_counter_ = other.sample_counter_;
    block_size_ = other.block_size_;
    deadline_offset_ = other.deadline_offset_;
}

size_t NupolsSegment::GetDelay() const
{
    return delay_;
}

void NupolsSegment::Process(CircularBuffer& input_buffer, CircularBuffer& output_buffer, size_t new_sample_count)
{
    sample_counter_ += new_sample_count;

    if (sample_counter_ == block_size_)
    {
        auto work_buffer = upols_.PrepareWorkBuffer();
        assert(work_buffer.size() == block_size_);
        input_buffer.Read(work_buffer, block_size_);

        upols_.Process(output_buffer_);
        assert(output_buffer_.size() == block_size_);

        output_buffer.Accumulate(output_buffer_, deadline_offset_);

        sample_counter_ = 0;
    }
}

void NupolsSegment::PrintPartition() const
{
    upols_.PrintPartition();
}

NUPOLS::NUPOLS(size_t block_size, std::span<const float> fir, PartitionStrategy strategy)
    : block_size_(block_size)
    , segments_()
{
    size_t circ_buffer_size = fir.size();
    if (circ_buffer_size % block_size != 0)
    {
        circ_buffer_size += block_size - (circ_buffer_size % block_size);
    }
    input_buffer_ = CircularBuffer(circ_buffer_size);
    output_buffer_ = CircularBuffer(circ_buffer_size);

    if (strategy == PartitionStrategy::kUpols)
    {
        segments_.emplace_back(block_size, block_size, 0, fir);
    }
    else if (strategy == PartitionStrategy::kMaxSegment)
    {
        size_t segment_count = std::ceil(static_cast<float>(fir.size()) / block_size);
        segments_.reserve(segment_count);
        for (size_t i = 0; i < segment_count; ++i)
        {
            size_t filter_block_size = std::min(block_size, fir.size() - i * block_size);
            segments_.emplace_back(block_size, block_size, i * block_size,
                                   fir.subspan(i * block_size, filter_block_size));
        }
    }
    else if (strategy == PartitionStrategy::kCanonical)
    {
        size_t segment_block_size = block_size;
        size_t fir_offset = 0;
        while (fir_offset < fir.size())
        {
            size_t segment_size = std::min(segment_block_size, fir.size() - fir_offset);
            segments_.emplace_back(block_size, segment_block_size, fir_offset, fir.subspan(fir_offset, segment_size));
            fir_offset += segment_size;
            segment_block_size *= 2;
        }
    }
    else if (strategy == PartitionStrategy::kGardner)
    {
        size_t segment_block_size = block_size;
        size_t fir_offset = 0;
        while (fir_offset < fir.size())
        {
            // max out at 8192 for no particular reason
            if (segment_block_size == 8192)
            {
                size_t segment_size = fir.size() - fir_offset;
                segments_.emplace_back(block_size, segment_block_size, fir_offset,
                                       fir.subspan(fir_offset, segment_size));
                fir_offset += segment_size;
                assert(fir_offset == fir.size());
            }
            else
            {
                // The original Gardner paper uses a factor of 2, but I found that using a factor of 4
                // gives better performance for my implementation.
                constexpr size_t kRepCount = 8;
                size_t segment_size = std::min(segment_block_size * kRepCount, fir.size() - fir_offset);
                segments_.emplace_back(block_size, segment_block_size, fir_offset,
                                       fir.subspan(fir_offset, segment_size));
                fir_offset += segment_size;
                segment_block_size *= kRepCount;
            }
        }
    }
    else
    {
        assert(false && "Invalid partition strategy");
    }
}

void NUPOLS::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == block_size_);
    assert(output.SampleCount() == block_size_);
    assert(input.ChannelCount() == 1);
    assert(output.ChannelCount() == 1);

    input_buffer_.Write(input.GetChannelSpan(0));
    input_buffer_.Advance(input.SampleCount());
    // Process each segment
    for (auto& segment : segments_)
    {
        segment.Process(input_buffer_, output_buffer_, input.SampleCount());
    }

    output_buffer_.Read(output.GetChannelSpan(0));
    output_buffer_.Clear(output.SampleCount());
    output_buffer_.Advance(output.SampleCount());
}

void NUPOLS::DumpInfo() const
{
    std::cout << "NUPOLS Info:" << std::endl;
    std::cout << "Block size: " << block_size_ << std::endl;
    std::cout << "Input buffer size: " << input_buffer_.Size() << std::endl;
    std::cout << "Number of segments: " << segments_.size() << std::endl;
    std::cout << "Segment delays:" << std::endl;
    for (size_t i = 0; i < segments_.size(); ++i)
    {
        const auto& segment = segments_[i];
        {
            std::cout << "    Segment #" << i << " delay: " << segment.GetDelay() << std::endl;
        }
    }

    for (size_t i = 0; i < segments_.size(); ++i)
    {
        segments_[i].PrintPartition();
    }
    std::cout << std::endl;
}
} // namespace sfFDN