#include "sffdn/audio_processor.h"

#include "pch.h"

namespace sfFDN
{

AudioProcessorChain::AudioProcessorChain(uint32_t block_size)
    : block_size_(block_size)
    , max_work_buffer_size_(block_size)
{
    // Initialize work buffers based on the block size
    work_buffer_a_.resize(max_work_buffer_size_);
    work_buffer_b_.resize(max_work_buffer_size_);
}

bool AudioProcessorChain::AddProcessor(std::unique_ptr<AudioProcessor>&& processor)
{
    if (processors_.empty())
    {
        processors_.emplace_back(std::move(processor));
    }
    else
    {
        if (processors_.back()->OutputChannelCount() != processor->InputChannelCount())
        {
            std::print(std::cerr,
                       "Output channel count of last processor does not match input channel count of new processor.");
            return false;
        }
        processors_.emplace_back(std::move(processor));
    }

    // Update the maximum work buffer size if necessary
    uint32_t work_buffer_size = processors_.back()->OutputChannelCount() * block_size_;
    if (work_buffer_size > max_work_buffer_size_)
    {
        max_work_buffer_size_ = work_buffer_size;
        work_buffer_a_.resize(max_work_buffer_size_);
        work_buffer_b_.resize(max_work_buffer_size_);
    }

    return true;
}

void AudioProcessorChain::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    if (processors_.empty())
    {
        return;
    }

    assert(input.ChannelCount() == processors_.front()->InputChannelCount());
    assert(output.ChannelCount() == processors_.back()->OutputChannelCount());
    assert(input.SampleCount() == output.SampleCount());
    assert(input.SampleCount() == block_size_);

    if (processors_.size() == 1)
    {
        processors_[0]->Process(input, output);
        return;
    }

    // Process the first audio processor
    AudioBuffer buffer_a(block_size_, processors_[0]->OutputChannelCount(), work_buffer_a_);
    processors_[0]->Process(input, buffer_a);

    std::span<float> ptr_a = work_buffer_a_;
    std::span<float> ptr_b = work_buffer_b_;

    // Process the rest of the audio processors in the chain
    for (auto i = 1; i < processors_.size() - 1; ++i)
    {
        AudioBuffer buffer_in(block_size_, processors_[i]->InputChannelCount(), ptr_a);
        AudioBuffer buffer_out(block_size_, processors_[i]->OutputChannelCount(), ptr_b);
        assert(processors_[i]->InputChannelCount() == buffer_in.ChannelCount());
        assert(processors_[i]->OutputChannelCount() == buffer_out.ChannelCount());
        processors_[i]->Process(buffer_in, buffer_out);
        std::swap(ptr_a, ptr_b);
    }

    // Process the last audio processor
    AudioBuffer buffer_in(block_size_, processors_.back()->InputChannelCount(), ptr_a);
    processors_.back()->Process(buffer_in, output);
}

uint32_t AudioProcessorChain::InputChannelCount() const
{
    if (processors_.empty())
    {
        return 0;
    }
    return processors_.front()->InputChannelCount();
}

uint32_t AudioProcessorChain::OutputChannelCount() const
{
    if (processors_.empty())
    {
        return 0;
    }
    return processors_.back()->OutputChannelCount();
}

void AudioProcessorChain::Clear()
{
    for (auto& processor : processors_)
    {
        processor->Clear();
    }
}

std::unique_ptr<AudioProcessor> AudioProcessorChain::Clone() const
{
    auto clone = std::make_unique<AudioProcessorChain>(block_size_);
    for (const auto& processor : processors_)
    {
        clone->AddProcessor(processor->Clone());
    }
    return clone;
}

} // namespace sfFDN