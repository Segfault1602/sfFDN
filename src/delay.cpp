#include "delay.h"

#include <cassert>
#include <iostream>

namespace sfFDN
{

Delay::Delay(uint32_t delay, uint32_t maxDelay)
{
    // Writing before reading allows delays from 0 to length-1.
    // If we want to allow a delay of maxDelay, we need a
    // delay-line of length = maxDelay+1.
    if (delay > maxDelay)
    {
        std::cerr << "Delay::Delay: maxDelay must be > than delay argument!\n";
        assert(false);
    }

    if ((maxDelay + 1) > buffer_.size())
        buffer_.resize(maxDelay + 1, 0.0);

    inPoint_ = 0;
    lastFrame_ = 0.0;
    this->SetDelay(delay);
}

Delay::~Delay()
{
}

void Delay::Clear()
{
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}

void Delay::SetMaximumDelay(unsigned long delay)
{
    if (delay < buffer_.size())
        return;
    buffer_.resize(delay + 1, 0.0);
}

void Delay::SetDelay(uint32_t delay)
{
    if (delay > buffer_.size() - 1)
    {
        assert(false);
        return;
    }

    // read chases write
    if (inPoint_ >= delay)
        outPoint_ = inPoint_ - delay;
    else
        outPoint_ = buffer_.size() + inPoint_ - delay;
    delay_ = delay;
}

float Delay::NextOut() const
{
    return buffer_[outPoint_];
}

float Delay::Tick(float input)
{
    buffer_[inPoint_] = input;
    inPoint_ = (inPoint_ + 1) % buffer_.size();

    // Read out next value
    lastFrame_ = buffer_[outPoint_];
    outPoint_ = (outPoint_ + 1) % buffer_.size();

    return lastFrame_;
}

void Delay::Process(const AudioBuffer input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // Delay only supports mono input

    AddNextInputs(input.GetChannelSpan(0));
    GetNextOutputs(output.GetChannelSpan(0));
}

void Delay::AddNextInputs(std::span<const float> input)
{
    // Check that we have enough space between the inPoint_ and outPoint_
    // to write the input data.
    uint32_t available_space =
        (outPoint_ > inPoint_) ? (outPoint_ - inPoint_) : (buffer_.size() - inPoint_ + outPoint_);
    if (available_space < input.size())
    {
        std::cerr << "Delay::Tick: Not enough space in buffer to write input data!\n";
        assert(false);
        return;
    }

    uint32_t end_point = inPoint_ + input.size();
    if (end_point > buffer_.size())
    {
        end_point = buffer_.size();
    }

    uint32_t size1 = end_point - inPoint_;
    uint32_t size2 = input.size() - size1;
    // Copy input to buffer
    std::copy(input.begin(), input.begin() + size1, buffer_.begin() + inPoint_);
    // Copy input to buffer
    if (size2 > 0)
    {
        std::copy(input.begin() + size1, input.end(), buffer_.begin());
    }

    inPoint_ = (inPoint_ + input.size()) % buffer_.size();
}

void Delay::GetNextOutputs(std::span<float> output)
{
    uint32_t outPoint = outPoint_;

    uint32_t endPoint = outPoint + output.size();
    if (endPoint < buffer_.size())
    {
        std::copy(buffer_.begin() + outPoint, buffer_.begin() + endPoint, output.begin());
    }
    else
    {
        uint32_t size1 = buffer_.size() - outPoint;
        uint32_t size2 = output.size() - size1;
        std::copy(buffer_.begin() + outPoint, buffer_.end(), output.begin());
        std::copy(buffer_.begin(), buffer_.begin() + size2, output.begin() + size1);
    }

    outPoint_ = endPoint % buffer_.size();
    lastFrame_ = output.back();
}

} // namespace sfFDN