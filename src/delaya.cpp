#include "delaya.h"

#include <cassert>
#include <iostream>

namespace fdn
{

DelayAllpass::DelayAllpass(float delay, unsigned long maxDelay)
{
    if (delay < 0.5)
    {
        std::cerr << "DelayAllpass::DelayAllpass: delay must be >= 0.5!" << std::endl;
        assert(false);
    }

    if (delay > (float)maxDelay)
    {
        std::cerr << "DelayAllpass::DelayAllpass: maxDelay must be > than delay argument!" << std::endl;
        assert(false);
    }

    // Writing before reading allows delays from 0 to length-1.
    if (maxDelay + 1 > buffer_.size())
        buffer_.resize(maxDelay + 1, 0.0);

    inPoint_ = 0;
    this->SetDelay(delay);
    apInput_ = 0.0;
    doNextOut_ = true;
    gain_ = 1.0;
    lastFrame_ = 0.0;
}

DelayAllpass::~DelayAllpass()
{
}

void DelayAllpass::Clear()
{
    for (unsigned int i = 0; i < buffer_.size(); i++)
        buffer_[i] = 0.0;
    lastFrame_ = 0.0;
    apInput_ = 0.0;
}

void DelayAllpass::SetMaximumDelay(unsigned long delay)
{
    if (delay < buffer_.size())
        return;
    buffer_.resize(delay + 1, 0.0);
}

void DelayAllpass::UpdateAlpha(float delay)
{
    float outPointer = inPoint_ - delay + 1.0; // outPoint chases inpoint

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

void DelayAllpass::SetDelay(float delay)
{
    unsigned long length = buffer_.size();
    if (delay + 1 > length)
    { // The value is too big.
        std::cerr << "DelayAllpass::setDelay: argument (" << delay << ") greater than maximum!" << std::endl;
        assert(false);
        return;
    }

    if (delay < 0.5)
    {
        std::cerr << "DelayAllpass::setDelay: argument (" << delay << ") less than 0.5 not possible!" << std::endl;
        assert(false);
        return;
    }

    UpdateAlpha(delay);
    delay_ = delay;
}

float DelayAllpass::NextOut()
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

float DelayAllpass::Tick(float input)
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

void DelayAllpass::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    auto input_span = input.GetChannelSpan(0);
    auto output_span = output.GetChannelSpan(0);

    for (size_t i = 0; i < input_span.size(); i++)
    {
        output_span[i] = Tick(input_span[i]);
    }
}

} // namespace fdn