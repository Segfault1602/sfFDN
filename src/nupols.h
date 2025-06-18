#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "audio_processor.h"
#include "circular_buffer.h"
#include "fft.h"
#include "upols.h"

namespace fdn
{

class NupolsSegment;

enum class PartitionStrategy
{
    kUpols,      // One segment, equivalent to UPOLS
    kMaxSegment, // Each segment has only one partition, all the same size, for testing
    kCanonical,  // [B][2B][4B][8B]... [2^n B]
    kGardner,    // [B|B][2B|2B][4B|4B]... [2^n B|2^n B]
};

class NUPOLS : public AudioProcessor
{
  public:
    NUPOLS(size_t block_size, std::span<const float> fir, PartitionStrategy strategy);
    ~NUPOLS() = default;

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void DumpInfo() const;

    size_t InputChannelCount() const override
    {
        return 1; // NUPOLS processes one channel at a time
    }
    size_t OutputChannelCount() const override
    {
        return 1; // NUPOLS processes one channel at a time
    }

  private:
    size_t block_size_;
    CircularBuffer input_buffer_;
    CircularBuffer output_buffer_;

    std::vector<NupolsSegment> segments_;
};

class NupolsSegment
{
  public:
    NupolsSegment(size_t parent_block_size, size_t block_size, size_t delay, std::span<const float> fir);
    ~NupolsSegment() = default;
    NupolsSegment(NupolsSegment&& other);

    size_t GetDelay() const;
    void Process(CircularBuffer& input_buffer, CircularBuffer& output_buffer, size_t new_sample_count);

    void PrintPartition() const;

  private:
    UPOLS upols_;
    const size_t delay_;
    size_t block_size_;

    size_t sample_counter_;
    size_t deadline_offset_;

    std::vector<float> output_buffer_;
};
} // namespace fdn