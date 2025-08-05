#include "sffdn/partitioned_convolver.h"

#include <cassert>
#include <print>

#include "array_math.h"
#include "circular_buffer.h"
#include "upols.h"

namespace sfFDN
{

class PartitionedConvolverSegment
{
  public:
    PartitionedConvolverSegment(size_t parent_block_size, size_t block_size, size_t delay, std::span<const float> fir);
    ~PartitionedConvolverSegment() = default;
    PartitionedConvolverSegment(PartitionedConvolverSegment&& other);

    size_t GetDelay() const;
    void Process(std::span<const float> input, CircularBuffer& output_buffer, size_t new_sample_count);

    void PrintPartition() const;

  private:
    UPOLS upols_;
    const size_t delay_;
    size_t block_size_;

    size_t deadline_offset_;

    std::vector<float> output_buffer_;
};

PartitionedConvolverSegment::PartitionedConvolverSegment(size_t parent_block_size, size_t block_size, size_t delay,
                                                         std::span<const float> fir)
    : upols_(block_size, fir)
    , delay_(delay)
    , block_size_(block_size)
{
    output_buffer_.resize(block_size, 0.f);

    size_t a = block_size / parent_block_size;
    assert(delay >= parent_block_size * (a - 1));
    deadline_offset_ = delay - parent_block_size * (a - 1);
}

PartitionedConvolverSegment::PartitionedConvolverSegment(PartitionedConvolverSegment&& other)
    : upols_(std::move(other.upols_))
    , delay_(other.delay_)
{
    output_buffer_ = std::move(other.output_buffer_);
    block_size_ = other.block_size_;
    deadline_offset_ = other.deadline_offset_;
}

size_t PartitionedConvolverSegment::GetDelay() const
{
    return delay_;
}

void PartitionedConvolverSegment::Process(std::span<const float> input, CircularBuffer& output_buffer,
                                          size_t new_sample_count)
{
    upols_.AddSamples(input);

    if (upols_.IsReady())
    {
        upols_.Process(output_buffer_);
        assert(output_buffer_.size() == block_size_);

        output_buffer.Accumulate(output_buffer_, deadline_offset_);
    }
}

void PartitionedConvolverSegment::PrintPartition() const
{
    upols_.PrintPartition();
}

class PartitionedConvolver::PartitionedConvolverImpl
{
  public:
    PartitionedConvolverImpl(size_t block_size, std::span<const float> fir)
        : block_size_(block_size)
        , segments_()
    {
        size_t circ_buffer_size = fir.size();
        if (circ_buffer_size % block_size != 0)
        {
            circ_buffer_size += block_size - (circ_buffer_size % block_size);
        }
        output_buffer_ = CircularBuffer(circ_buffer_size);

        size_t segment_block_size = block_size;
        size_t fir_offset = 0;
        while (fir_offset < fir.size())
        {
            // max out at 8192 for no particular reason
            if (segment_block_size == 8192)
            {
                size_t segment_size = fir.size() - fir_offset;
                segments_.emplace_back(std::make_unique<PartitionedConvolverSegment>(
                    block_size, segment_block_size, fir_offset, fir.subspan(fir_offset, segment_size)));
                fir_offset += segment_size;
                assert(fir_offset == fir.size());
            }
            else
            {
                // The original Gardner paper uses a factor of 2, but I found that using a factor of 4
                // gives better performance for my implementation.
                constexpr size_t kRepCount = 8;
                size_t segment_size = std::min(segment_block_size * kRepCount, fir.size() - fir_offset);
                segments_.emplace_back(std::make_unique<PartitionedConvolverSegment>(
                    block_size, segment_block_size, fir_offset, fir.subspan(fir_offset, segment_size)));
                fir_offset += segment_size;
                segment_block_size *= kRepCount;
            }
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output)
    {
        assert(input.SampleCount() == block_size_);
        assert(output.SampleCount() == block_size_);
        assert(input.ChannelCount() == 1);
        assert(output.ChannelCount() == 1);

        // Process each segment
        for (auto& segment : segments_)
        {
            segment->Process(input.GetChannelSpan(0), output_buffer_, input.SampleCount());
        }

        output_buffer_.Advance(output.SampleCount());
        output_buffer_.Read(output.GetChannelSpan(0), true);
        // output_buffer_.Clear(output.SampleCount());
    }

    void DumpInfo() const
    {
        std::println("PartitionedConvolver Info:");
        std::println("Block size: {}", block_size_);
        std::println("Number of segments: {}", segments_.size());
        std::println("Segment delays:");
        for (size_t i = 0; i < segments_.size(); ++i)
        {
            const auto& segment = segments_[i];
            {
                std::println("    Segment #{} delay: {}", i, segment->GetDelay());
            }
        }

        for (size_t i = 0; i < segments_.size(); ++i)
        {
            segments_[i]->PrintPartition();
        }
        std::println();
    }

  private:
    size_t block_size_;
    CircularBuffer output_buffer_;

    std::vector<std::unique_ptr<PartitionedConvolverSegment>> segments_;
};

PartitionedConvolver::PartitionedConvolver(size_t block_size, std::span<const float> fir)
{
    impl_ = std::make_unique<PartitionedConvolverImpl>(block_size, fir);
}

PartitionedConvolver::~PartitionedConvolver() = default;

void PartitionedConvolver::Process(const AudioBuffer& input, AudioBuffer& output)
{
    impl_->Process(input, output);
}

void PartitionedConvolver::DumpInfo() const
{
    impl_->DumpInfo();
}
} // namespace sfFDN