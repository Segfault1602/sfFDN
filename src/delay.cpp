#include "sffdn/delay.h"

#include "pch.h"
#include <algorithm>
#include <cstdint>

namespace
{
uint32_t fast_mod(uint32_t input, uint32_t ceil)
{
    return input >= ceil ? input % ceil : input;
}
} // namespace

namespace sfFDN
{

Delay::Delay(uint32_t delay, uint32_t maxDelay)
    : in_point_(0)
    , out_point_(0)
    , delay_(0)
    , last_frame_(0.0f)
{
    if (delay > maxDelay)
    {
        std::cerr << "Delay::Delay: maxDelay must be > than delay argument!\n";
        assert(false);
    }

    if ((maxDelay + 1) > buffer_.size())
    {
        buffer_.resize(maxDelay + 1, 0.0);
    }

    in_point_ = 0;
    last_frame_ = 0.0;
    this->SetDelay(delay);
}

Delay::~Delay()
{
}

void Delay::Clear()
{
    std::ranges::fill(buffer_, 0.0f);
}

void Delay::SetMaximumDelay(uint32_t delay)
{
    if (delay < buffer_.size())
    {
        return;
    }
    buffer_.resize(delay + 1, 0.0);
}

uint32_t Delay::GetMaximumDelay() const
{
    return buffer_.size() - 1;
}

void Delay::SetDelay(uint32_t delay)
{
    if (delay == delay_)
    {
        return;
    }

    if (delay > buffer_.size() - 1)
    {
        assert(false);
        return;
    }

    // read chases write
    if (in_point_ >= delay)
    {
        out_point_ = in_point_ - delay;
    }
    else
    {
        out_point_ = buffer_.size() + in_point_ - delay;
    }
    delay_ = delay;
}

float Delay::NextOut() const
{
    return buffer_[out_point_];
}

float Delay::Tick(float input)
{
    buffer_[in_point_] = input;
    in_point_ = fast_mod(in_point_ + 1, buffer_.size());

    // Read out next value
    last_frame_ = buffer_[out_point_];
    out_point_ = fast_mod(out_point_ + 1, buffer_.size());

    return last_frame_;
}

float Delay::TapOut(uint32_t tap) const
{
    if (tap >= buffer_.size())
    {
        std::cerr << "Delay::TapOut: Tap point exceeds buffer size!\n";
        assert(false);
        return 0.0f;
    }

    uint32_t tap_point = (in_point_ + buffer_.size() - tap - 1) % buffer_.size();
    return buffer_[tap_point];
}

void Delay::Process(const AudioBuffer input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // Delay only supports mono input

    if (AddNextInputs(input.GetChannelSpan(0)))
    {
        GetNextOutputs(output.GetChannelSpan(0));
    }
    else
    {
        // We could not add all input samples at once, so just process samples one by one.
        auto input_span = input.GetChannelSpan(0);
        auto output_span = output.GetChannelSpan(0);
        for (auto i = 0; i < input.SampleCount(); ++i)
        {
            output_span[i] = Tick(input_span[i]);
        }
    }
}

bool Delay::AddNextInputs(std::span<const float> input)
{
    std::span<float> buffer_span = buffer_;

    std::span<float> buffer_1{};
    std::span<float> buffer_2{};

    // Two scenarios:
    // 1. The out pointer is after the in pointer
    //      In this case we have all the space between in_point_ and out_point_ that we can write to.
    // 2. The in pointer is after the out pointer
    //      The write region wraps around the buffer. We have two regions to consider. The first region is from
    //      in_point_ to the end of the buffer, and the second region is from the beginning of the buffer to out_point_.
    if (out_point_ > in_point_)
    {
        buffer_1 = buffer_span.subspan(in_point_, out_point_ - in_point_);
        buffer_2 = {};
    }
    else
    {
        buffer_1 = buffer_span.subspan(in_point_, buffer_span.size() - in_point_);
        buffer_2 = buffer_span.subspan(0, out_point_);
    }

    // Check that we have enough space between the inPoint_ and outPoint_
    // to write the input data.
    uint32_t available_space = buffer_1.size() + buffer_2.size();
    if (available_space < input.size())
    {
        std::cerr << "Delay::Tick: Not enough space in buffer to write input data!\n";
        assert(false);
        return false;
    }

    if (input.size() <= buffer_1.size())
    {
        // All input fits in the first region
        std::ranges::copy(input, buffer_1.begin());
    }
    else
    {
        std::ranges::copy(input.first(buffer_1.size()), buffer_1.begin());
        std::ranges::copy(input.subspan(buffer_1.size()), buffer_2.begin());
    }

    in_point_ = fast_mod(in_point_ + input.size(), buffer_.size());
    return true;
}

void Delay::GetNextOutputs(std::span<float> output)
{
    std::span<float> buffer_span = buffer_;

    std::span<float> buffer_1{};
    std::span<float> buffer_2{};

    // Two scenarios:
    // 1. The out pointer is after the in pointer
    //      The read region wraps around the buffer. We have two regions to consider. The first region is from
    //      out_point to the end of the buffer, and the second region is from the beginning of the buffer to in_point_.
    // 2. The in pointer is after the out pointer
    //      In this case we have all the space between in_point_ and out_point_ that we can read from.
    if (out_point_ > in_point_)
    {
        buffer_1 = buffer_span.subspan(out_point_);
        buffer_2 = buffer_span.subspan(0, in_point_);
    }
    else
    {
        buffer_1 = buffer_span.subspan(out_point_, in_point_ - out_point_);
        buffer_2 = {};
    }

    // Check that we have enough data to read from
    uint32_t available_space = buffer_1.size() + buffer_2.size();
    if (available_space < output.size())
    {
        std::cerr << "Delay::GetNextOutputs: Not enough data in buffer to read output data!\n";
        assert(false);
        return;
    }

    if (buffer_1.size() >= output.size())
    {
        std::ranges::copy(buffer_1.first(output.size()), output.begin());
    }
    else
    {
        std::ranges::copy(buffer_1, output.begin());
        std::ranges::copy(buffer_2.first(output.size() - buffer_1.size()), output.subspan(buffer_1.size()).begin());
    }

    out_point_ = (out_point_ + output.size()) % buffer_.size();
    last_frame_ = output.back();
}

} // namespace sfFDN