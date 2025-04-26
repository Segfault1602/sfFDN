/***************************************************/
/*! \class Delay
    \brief STK non-interpolating delay line class.

    This class implements a non-interpolating digital delay-line.  If
    the delay and maximum length are not specified during
    instantiation, a fixed maximum length of 4095 and a delay of zero
    is set.

    A non-interpolating delay line is typically used in fixed
    delay-length applications, such as for reverberation.

    by Perry R. Cook and Gary P. Scavone, 1995--2023.
*/
/***************************************************/

#include "delay.h"

#include <cassert>
#include <iostream>

namespace fdn
{

Delay::Delay(unsigned long delay, unsigned long maxDelay)
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

void Delay::SetDelay(unsigned long delay)
{
    if (delay > buffer_.size() - 1)
    { // The value is too big.
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

float Delay::Tick(float input)
{
    buffer_[inPoint_++] = input;

    // Check for end condition
    if (inPoint_ == buffer_.size())
        inPoint_ = 0;

    // Read out next value
    lastFrame_ = buffer_[outPoint_++];

    if (outPoint_ == buffer_.size())
        outPoint_ = 0;

    return lastFrame_;
}

void Delay::Tick(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());

    // TODO: this needs to be tested

    // Check that we have enough space between the inPoint_ and outPoint_
    // to write the input data.
    size_t available_space = (outPoint_ > inPoint_) ? (outPoint_ - inPoint_) : (buffer_.size() - inPoint_ + outPoint_);
    if (available_space < input.size())
    {
        std::cerr << "Delay::Tick: Not enough space in buffer to write input data!\n";
        assert(false);
        return;
    }

    size_t end_point = inPoint_ + input.size();
    if (end_point > buffer_.size())
    {
        end_point = buffer_.size();
    }

    size_t size1 = end_point - inPoint_;
    size_t size2 = input.size() - size1;
    // Copy input to buffer
    std::copy(input.begin(), input.begin() + size1, buffer_.begin() + inPoint_);
    // Copy input to buffer
    if (size2 > 0)
    {
        std::copy(input.begin() + size1, input.end(), buffer_.begin());
    }

    inPoint_ += input.size();
    while (inPoint_ >= buffer_.size())
    {
        inPoint_ -= buffer_.size();
    }

    // read out next values
    size_t outPoint = outPoint_;
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = buffer_[outPoint++];
        if (outPoint == buffer_.size())
            outPoint = 0;
    }
    outPoint_ = outPoint;

    lastFrame_ = output.back();
}

} // namespace fdn