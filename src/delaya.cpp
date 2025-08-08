#include "sffdn/delaya.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <print>

namespace sfFDN
{

DelayAllpass::DelayAllpass(float delay, uint32_t maxDelay)
    : in_point_(0)
    , out_point_(0)
    , delay_(0.0f)
    , alpha_(0.0f)
    , coeff_(0.0f)
    , next_output_(0.0f)
{
    if (delay < 0.5f)
    {
        std::println(std::cerr, "DelayAllpass::DelayAllpass: delay must be >= 0.5!");
        assert(false);
        delay = 0.5f; // Set to minimum valid value
    }

    if (delay > (float)maxDelay)
    {
        std::println(std::cerr, "DelayAllpass::DelayAllpass: maxDelay must be > than delay argument!");
        assert(false);
        delay = maxDelay - 1; // Set to maximum valid value
    }

    // Writing before reading allows delays from 0 to length-1.
    if (maxDelay + 1 > buffer_.size())
    {
        buffer_.resize(maxDelay + 1, 0.0);
    }

    in_point_ = 0;
    this->SetDelay(delay);
    ap_input_ = 0.0;
    do_next_out_ = true;
    gain_ = 1.0;
    last_frame_ = 0.0;
}

void DelayAllpass::Clear()
{
    std::ranges::fill(buffer_, 0.0f);
    last_frame_ = 0.0;
    ap_input_ = 0.0;
}

void DelayAllpass::SetMaximumDelay(uint32_t delay)
{
    if (delay < buffer_.size())
    {
        return;
    }
    buffer_.resize(delay + 1, 0.0);
}

void DelayAllpass::UpdateAlpha(float delay)
{
    float outPointer = in_point_ - delay + 1.0f;

    uint32_t length = buffer_.size();
    while (outPointer < 0)
    {
        outPointer += length;
    }

    out_point_ = static_cast<uint32_t>(outPointer);
    if (out_point_ == length)
    {
        out_point_ = 0;
    }
    alpha_ = 1.0f + out_point_ - outPointer;

    if (alpha_ < 0.5f)
    {
        // The optimal range for alpha is about 0.5 - 1.5 in order to
        // achieve the flattest phase delay response.
        out_point_ += 1;
        if (out_point_ >= length)
        {
            out_point_ -= length;
        }
        alpha_ += 1.0f;
    }

    coeff_ = (1.0f - alpha_) / (1.0f + alpha_);
}

void DelayAllpass::SetDelay(float delay)
{
    uint32_t length = buffer_.size();
    if (delay + 1 > length)
    { // The value is too big.
        std::println(std::cerr, "DelayAllpass::setDelay: argument ({}) greater than maximum!", delay);
        assert(false);
        return;
    }

    if (delay < 0.5)
    {
        std::println(std::cerr, "DelayAllpass::setDelay: argument ({}) less than 0.5 not possible!", delay);
        assert(false);
        return;
    }

    UpdateAlpha(delay);
    delay_ = delay;
}

float DelayAllpass::NextOut()
{
    if (do_next_out_)
    {
        // Do allpass interpolation delay.
        next_output_ = -coeff_ * last_frame_;
        next_output_ += ap_input_ + (coeff_ * buffer_[out_point_]);
        do_next_out_ = false;
    }

    return next_output_;
}

float DelayAllpass::Tick(float input)
{
    buffer_[in_point_++] = input * gain_;

    // Increment input pointer modulo length.
    if (in_point_ == buffer_.size())
    {
        in_point_ = 0;
    }

    last_frame_ = NextOut();
    do_next_out_ = true;

    // Save the allpass input and increment modulo length.
    ap_input_ = buffer_[out_point_++];
    if (out_point_ == buffer_.size())
    {
        out_point_ = 0;
    }

    return last_frame_;
}

void DelayAllpass::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    auto input_span = input.GetChannelSpan(0);
    auto output_span = output.GetChannelSpan(0);

    for (auto i = 0; i < input_span.size(); i++)
    {
        output_span[i] = Tick(input_span[i]);
    }
}

} // namespace sfFDN