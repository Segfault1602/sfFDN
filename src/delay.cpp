#include "sffdn/delay.h"

#include "pch.h"

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
    in_point_ = (in_point_ + 1) % buffer_.size();

    // Read out next value
    last_frame_ = buffer_[out_point_];
    out_point_ = (out_point_ + 1) % buffer_.size();

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
    // Check that we have enough space between the inPoint_ and outPoint_
    // to write the input data.
    uint32_t available_space =
        (out_point_ > in_point_) ? (out_point_ - in_point_) : (buffer_.size() - in_point_ + out_point_);
    if (available_space < input.size())
    {
        std::cerr << "Delay::Tick: Not enough space in buffer to write input data!\n";
        assert(false);
        return false;
    }

    uint32_t end_point = in_point_ + input.size();
    end_point = std::min(end_point, static_cast<uint32_t>(buffer_.size()));

    uint32_t size1 = end_point - in_point_;
    uint32_t size2 = input.size() - size1;
    // Copy input to buffer
    std::copy(input.begin(), input.begin() + size1, buffer_.begin() + in_point_);
    // Copy input to buffer
    if (size2 > 0)
    {
        std::copy(input.begin() + size1, input.end(), buffer_.begin());
    }

    in_point_ = (in_point_ + input.size()) % buffer_.size();
    return true;
}

void Delay::GetNextOutputs(std::span<float> output)
{
    uint32_t outPoint = out_point_;

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

    out_point_ = endPoint % buffer_.size();
    last_frame_ = output.back();
}

} // namespace sfFDN