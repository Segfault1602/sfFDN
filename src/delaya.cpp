/***************************************************/
/*! \class DelayA
    \brief STK allpass interpolating delay line class.

    This class implements a fractional-length digital delay-line using
    a first-order allpass filter.  If the delay and maximum length are
    not specified during instantiation, a fixed maximum length of 4095
    and a delay of zero is set.

    An allpass filter has unity magnitude gain but variable phase
    delay properties, making it useful in achieving fractional delays
    without affecting a signal's frequency magnitude response.  In
    order to achieve a maximally flat phase delay response, the
    minimum delay possible in this implementation is limited to a
    value of 0.5.

    by Perry R. Cook and Gary P. Scavone, 1995--2023.
*/
/***************************************************/

#include "delaya.h"

#include <cassert>
#include <iostream>

namespace fdn
{

DelayA::DelayA(float delay, unsigned long maxDelay)
{
    if (delay < 0.5)
    {
        std::cerr << "DelayA::DelayA: delay must be >= 0.5!" << std::endl;
        assert(false);
    }

    if (delay > (float)maxDelay)
    {
        std::cerr << "DelayA::DelayA: maxDelay must be > than delay argument!" << std::endl;
        assert(false);
    }

    // Writing before reading allows delays from 0 to length-1.
    if (maxDelay + 1 > buffer_.size())
        buffer_.resize(maxDelay + 1, 0.0);

    inPoint_ = 0;
    fake_in_point_ = 0;
    this->SetDelay(delay);
    apInput_ = 0.0;
    doNextOut_ = true;
    gain_ = 1.0;
    lastFrame_ = 0.0;
}

DelayA::~DelayA()
{
}

void DelayA::Clear()
{
    for (unsigned int i = 0; i < buffer_.size(); i++)
        buffer_[i] = 0.0;
    lastFrame_ = 0.0;
    apInput_ = 0.0;
}

void DelayA::SetMaximumDelay(unsigned long delay)
{
    if (delay < buffer_.size())
        return;
    buffer_.resize(delay + 1, 0.0);
}

void DelayA::UpdateAlpha(float delay)
{
    float outPointer = fake_in_point_ - delay + 1.0; // outPoint chases inpoint

    unsigned long length = buffer_.size();
    while (outPointer < 0)
    {
        outPointer += length; // modulo maximum length
    }

    outPoint_ = (long)outPointer; // integer part
    if (outPoint_ == length)
    {
        outPoint_ = 0;
    }
    alpha_ = 1.0 + outPoint_ - outPointer; // fractional part

    if (alpha_ < 0.5)
    {
        // The optimal range for alpha is about 0.5 - 1.5 in order to
        // achieve the flattest phase delay response.
        outPoint_ += 1;
        if (outPoint_ >= length)
            outPoint_ -= length;
        alpha_ += (float)1.0;
    }

    coeff_ = (1.0 - alpha_) / (1.0 + alpha_); // coefficient for allpass
}

void DelayA::SetDelay(float delay)
{
    unsigned long length = buffer_.size();
    if (delay + 1 > length)
    { // The value is too big.
        std::cerr << "DelayA::setDelay: argument (" << delay << ") greater than maximum!" << std::endl;
        assert(false);
        return;
    }

    if (delay < 0.5)
    {
        std::cerr << "DelayA::setDelay: argument (" << delay << ") less than 0.5 not possible!" << std::endl;
        assert(false);
        return;
    }

    UpdateAlpha(delay);
    delay_ = delay;
}

float DelayA::NextOut(void)
{
    if (doNextOut_)
    {
        // Do allpass interpolation delay.
        nextOutput_ = -coeff_ * lastFrame_;
        nextOutput_ += apInput_ + (coeff_ * buffer_[outPoint_]);
        doNextOut_ = false;
    }

    return nextOutput_;
}

float DelayA::Tick(float input)
{
    buffer_[inPoint_++] = input * gain_;

    // Increment input pointer modulo length.
    if (inPoint_ == buffer_.size())
        inPoint_ = 0;

    lastFrame_ = NextOut();
    doNextOut_ = true;

    // Save the allpass input and increment modulo length.
    apInput_ = buffer_[outPoint_++];
    if (outPoint_ == buffer_.size())
        outPoint_ = 0;

    return lastFrame_;
}

void DelayA::Tick(std::span<float> input, std::span<float> output)
{
    assert(input.size() == output.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = Tick(input[i]);
    }
}

void DelayA::AddNextInput(float input)
{
    // assert(inPoint_ != outPoint_);
    buffer_[inPoint_++] = input * gain_;

    // Increment input pointer modulo length.
    if (inPoint_ == buffer_.size())
        inPoint_ = 0;
}

float DelayA::GetNextOutput()
{
    // assert(inPoint_ != outPoint_);
    lastFrame_ = NextOut();
    doNextOut_ = true;

    // Save the allpass input and increment modulo length.
    apInput_ = buffer_[outPoint_++];
    if (outPoint_ == buffer_.size())
        outPoint_ = 0;

    ++fake_in_point_;
    if (fake_in_point_ == buffer_.size())
        fake_in_point_ = 0;

    return lastFrame_;
}

void DelayA::AddNextInputs(std::span<const float> input)
{
    assert(input.size() < delay_);

    for (size_t i = 0; i < input.size(); i++)
    {
        assert(inPoint_ != outPoint_);
        buffer_[inPoint_++] = input[i] * gain_;

        // Increment input pointer modulo length.
        if (inPoint_ == buffer_.size())
            inPoint_ = 0;
    }
}

void DelayA::GetNextOutputs(std::span<float> output)
{
    assert(output.size() < delay_);

    for (size_t i = 0; i < output.size(); i++)
    {
        output[i] = GetNextOutput();
    }
}

} // namespace fdn