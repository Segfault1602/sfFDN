#include "sffdn/partitioned_convolver.h"

#include "circular_buffer.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "upols.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
uint32_t ResolveRepCount(uint32_t block_size, size_t fir_size, uint32_t requested_rep_count)
{
    if (requested_rep_count != 0)
    {
        return requested_rep_count;
    }
#if defined(__APPLE__) && defined(__aarch64__)
    return fir_size > 48u * static_cast<size_t>(block_size) ? 16u : 8u;
#else
    return 8u;
#endif
}

class PartitionedConvolverSegment
{
  public:
    PartitionedConvolverSegment(uint32_t parent_block_size, uint32_t block_size, uint32_t delay,
                                std::span<const float> fir);

    uint32_t GetDelay() const;
    void Process(std::span<const float> input, sfFDN::CircularBuffer& output_buffer) noexcept SFFDN_NONBLOCKING;

    void PrintPartition() const;
    std::string GetShortInfo() const;
    void Clear();

  private:
    sfFDN::UPOLS upols_;
    uint32_t delay_;
    uint32_t deadline_offset_;
    std::vector<float> output_buffer_;

    int current_deadline_{0};
};

PartitionedConvolverSegment::PartitionedConvolverSegment(uint32_t parent_block_size, uint32_t block_size,
                                                         uint32_t delay, std::span<const float> fir)
    : delay_(delay)
    , current_deadline_(delay)
{
    if (!upols_.Initialize(block_size, fir))
    {
        throw std::runtime_error("PartitionedConvolverSegment: Failed to initialize UPOLS");
    }
    output_buffer_.resize(block_size, 0.f);

    const uint32_t a = block_size / parent_block_size;
    assert(delay >= parent_block_size * (a - 1));
    deadline_offset_ = delay - parent_block_size * (a - 1);
}

uint32_t PartitionedConvolverSegment::GetDelay() const
{
    return delay_;
}

void PartitionedConvolverSegment::Process(std::span<const float> input,
                                          sfFDN::CircularBuffer& output_buffer) noexcept SFFDN_NONBLOCKING
{
    upols_.AddSamples(input);

    if (upols_.IsReady())
    {
        upols_.Process(output_buffer_);
        output_buffer.Accumulate(output_buffer_, current_deadline_);
        current_deadline_ = delay_;
    }
    else
    {
        current_deadline_ -= input.size();
    }
}

void PartitionedConvolverSegment::PrintPartition() const
{
    upols_.PrintPartition();
}

std::string PartitionedConvolverSegment::GetShortInfo() const
{
    return upols_.GetShortInfo();
}

void PartitionedConvolverSegment::Clear()
{
    upols_.Clear();
    std::ranges::fill(output_buffer_, 0.f);
}
} // namespace

namespace sfFDN
{

class PartitionedConvolver::PartitionedConvolverImpl
{
  public:
    PartitionedConvolverImpl(uint32_t block_size, std::span<const float> fir, uint32_t rep_count)
        : block_size_(block_size)
        , rep_count_(ResolveRepCount(block_size, fir.size(), rep_count))
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
            // max out at 16384 for no particular reason
            if (segment_block_size >= 16384)
            {
                segment_block_size = 16384;
                const uint32_t segment_size = fir.size() - fir_offset;
                segments_.emplace_back(block_size, segment_block_size, fir_offset,
                                       fir.subspan(fir_offset, segment_size));
                fir_offset += segment_size;
                assert(fir_offset == fir.size());
            }
            else
            {
                const uint32_t segment_size =
                    std::min(segment_block_size * rep_count_, static_cast<uint32_t>(fir.size()) - fir_offset);
                segments_.emplace_back(block_size, segment_block_size, fir_offset,
                                       fir.subspan(fir_offset, segment_size));
                fir_offset += segment_size;
                segment_block_size *= rep_count_;
            }
        }
    }

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
    {
        assert(input.SampleCount() == block_size_);
        assert(output.SampleCount() == block_size_);
        assert(input.ChannelCount() == 1);
        assert(output.ChannelCount() == 1);

        // Process each segment
        for (auto& segment : segments_)
        {
            segment.Process(input.GetChannelSpan(0), output_buffer_);
        }

        output_buffer_.Advance(output.SampleCount());
        output_buffer_.Read(output.GetChannelSpan(0), true);
    }

    uint32_t GetBlockSize() const
    {
        return block_size_;
    }

    void DumpInfo() const
    {
        std::println("PartitionedConvolver Info:");
        std::println("Block size: {}", block_size_);
        std::println("Number of segments: {}", segments_.size());
        std::println("Segment delays:");
        for (auto i = 0u; i < segments_.size(); ++i)
        {
            const auto& segment = segments_[i];
            {
                std::println("    Segment #{} delay: {}", i, segment.GetDelay());
            }
        }

        for (const auto& segment : segments_)
        {
            segment.PrintPartition();
        }
        std::println("");
    }

    std::string GetShortInfo() const
    {
        std::stringstream ss;
        ss << "[";
        for (auto i = 0u; i < segments_.size(); ++i)
        {
            if (i > 0)
            {
                ss << ", ";
            }
            ss << segments_[i].GetShortInfo();
        }
        ss << "]";
        return ss.str();
    }

    void Clear()
    {
        output_buffer_.Clear();
        for (auto& segment : segments_)
        {
            segment.Clear();
        }
    }

    std::unique_ptr<PartitionedConvolverImpl> Clone() const
    {
        return std::make_unique<PartitionedConvolverImpl>(block_size_, fir_, rep_count_);
    }

    nlohmann::json ToJson() const
    {
        nlohmann::json j;
        j["type"] = "PartitionedConvolver";
        j["block_size"] = block_size_;
        j["rep_count"] = rep_count_;
        j["fir_size"] = fir_.size();
        // The filter is potentially large, so it's not included in the JSON.
        return j;
    }

  private:
    uint32_t block_size_;
    CircularBuffer output_buffer_;
    uint32_t rep_count_;

    std::vector<PartitionedConvolverSegment> segments_;

    std::vector<float> fir_; // Store the FIR coefficients for cloning and, eventually, serializing
};

PartitionedConvolver::PartitionedConvolver(uint32_t block_size, std::span<const float> fir, uint32_t rep_count)
{
    impl_ = std::make_unique<PartitionedConvolverImpl>(block_size, fir, rep_count);
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

void PartitionedConvolver::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    impl_->Process(input, output);
}

uint32_t PartitionedConvolver::GetBlockSize() const
{
    return impl_->GetBlockSize();
}

void PartitionedConvolver::DumpInfo() const
{
    impl_->DumpInfo();
}

std::string PartitionedConvolver::GetShortInfo() const
{
    return impl_->GetShortInfo();
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