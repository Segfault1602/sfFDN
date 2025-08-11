#include "sffdn/partitioned_convolver.h"

#include <cassert>
#include <print>

#include "circular_buffer.h"
#include "upols.h"

namespace sfFDN
{

class PartitionedConvolverSegment
{
  public:
    PartitionedConvolverSegment(uint32_t parent_block_size, uint32_t block_size, uint32_t delay,
                                std::span<const float> fir);

    uint32_t GetDelay() const;
    void Process(std::span<const float> input, CircularBuffer& output_buffer);

    void PrintPartition() const;
    void Clear();

  private:
    UPOLS upols_;
    uint32_t delay_;
    uint32_t deadline_offset_;
    std::vector<float> output_buffer_;
};

PartitionedConvolverSegment::PartitionedConvolverSegment(uint32_t parent_block_size, uint32_t block_size,
                                                         uint32_t delay, std::span<const float> fir)
    : upols_(block_size, fir)
    , delay_(delay)
{
    output_buffer_.resize(block_size, 0.f);

    uint32_t a = block_size / parent_block_size;
    assert(delay >= parent_block_size * (a - 1));
    deadline_offset_ = delay - parent_block_size * (a - 1);
}

uint32_t PartitionedConvolverSegment::GetDelay() const
{
    return delay_;
}

void PartitionedConvolverSegment::Process(std::span<const float> input, CircularBuffer& output_buffer)
{
    upols_.AddSamples(input);

    if (upols_.IsReady())
    {
        upols_.Process(output_buffer_);
        output_buffer.Accumulate(output_buffer_, deadline_offset_);
    }
}

void PartitionedConvolverSegment::PrintPartition() const
{
    upols_.PrintPartition();
}

void PartitionedConvolverSegment::Clear()
{
    upols_.Clear();
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.f);
}

class PartitionedConvolver::PartitionedConvolverImpl
{
  public:
    PartitionedConvolverImpl(uint32_t block_size, std::span<const float> fir)
        : block_size_(block_size)
        , fir_(fir.begin(), fir.end())
    {
        uint32_t circ_buffer_size = fir.size();
        if (circ_buffer_size % block_size != 0)
        {
            circ_buffer_size += block_size - (circ_buffer_size % block_size);
        }
        output_buffer_ = CircularBuffer(circ_buffer_size);

        uint32_t segment_block_size = block_size;
        uint32_t fir_offset = 0;
        while (fir_offset < fir.size())
        {
            // max out at 8192 for no particular reason
            if (segment_block_size == 8192)
            {
                uint32_t segment_size = fir.size() - fir_offset;
                segments_.emplace_back(std::make_unique<PartitionedConvolverSegment>(
                    block_size, segment_block_size, fir_offset, fir.subspan(fir_offset, segment_size)));
                fir_offset += segment_size;
                assert(fir_offset == fir.size());
            }
            else
            {
                // The original Gardner paper uses a factor of 2, but I found that using a factor of 4
                // gives better performance for my implementation.
                constexpr uint32_t kRepCount = 8;
                uint32_t segment_size =
                    std::min(segment_block_size * kRepCount, static_cast<uint32_t>(fir.size()) - fir_offset);
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
            segment->Process(input.GetChannelSpan(0), output_buffer_);
        }

        output_buffer_.Advance(output.SampleCount());
        output_buffer_.Read(output.GetChannelSpan(0), true);
    }

    void DumpInfo() const
    {
        std::println("PartitionedConvolver Info:");
        std::println("Block size: {}", block_size_);
        std::println("Number of segments: {}", segments_.size());
        std::println("Segment delays:");
        for (auto i = 0; i < segments_.size(); ++i)
        {
            const auto& segment = segments_[i];
            {
                std::println("    Segment #{} delay: {}", i, segment->GetDelay());
            }
        }

        for (auto i = 0; i < segments_.size(); ++i)
        {
            segments_[i]->PrintPartition();
        }
        std::println("");
    }

    void Clear()
    {
        output_buffer_.Clear();
        for (auto& segment : segments_)
        {
            segment->Clear();
        }
    }

    std::unique_ptr<PartitionedConvolverImpl> Clone() const
    {
        return std::make_unique<PartitionedConvolverImpl>(block_size_, fir_);
    }

  private:
    uint32_t block_size_;
    CircularBuffer output_buffer_;

    std::vector<std::unique_ptr<PartitionedConvolverSegment>> segments_;

    std::vector<float> fir_; // Store the FIR coefficients for cloning and, eventually, serializing
};

PartitionedConvolver::PartitionedConvolver(uint32_t block_size, std::span<const float> fir)
{
    impl_ = std::make_unique<PartitionedConvolverImpl>(block_size, fir);
}

PartitionedConvolver::~PartitionedConvolver() = default;

PartitionedConvolver::PartitionedConvolver(PartitionedConvolver&& other) noexcept
    : impl_(std::move(other.impl_))
{
}

PartitionedConvolver& PartitionedConvolver::operator=(PartitionedConvolver&& other) noexcept
{
    impl_ = std::move(other.impl_);
    return *this;
}

void PartitionedConvolver::Process(const AudioBuffer& input, AudioBuffer& output)
{
    impl_->Process(input, output);
}

void PartitionedConvolver::DumpInfo() const
{
    impl_->DumpInfo();
}

void PartitionedConvolver::Clear()
{
    impl_->Clear();
}

std::unique_ptr<AudioProcessor> PartitionedConvolver::Clone() const
{
    auto clone = std::unique_ptr<PartitionedConvolver>(new PartitionedConvolver());
    clone->impl_ = impl_->Clone();
    return clone;
}

} // namespace sfFDN