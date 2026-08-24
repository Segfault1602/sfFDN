#include "sffdn/delay_interp.h"

#include "sffdn/audio_buffer.h"

#include "json_helper.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <ranges>

namespace
{
template <size_t N>
std::array<float, N + 1> GetLagrangeCoefficients(float delay)
{
    std::array<float, N + 1> coeffs{0.f};
    std::fill(coeffs.begin(), coeffs.end(), 1.0f);
    for (size_t k = 0; k <= N; ++k)
    {
        for (size_t j = 0; j <= N; ++j)
        {
            if (j != k)
            {
                coeffs[j] =
                    coeffs[j] * (delay - static_cast<float>(k)) / (static_cast<float>(j) - static_cast<float>(k));
            }
        }
    }

    return coeffs;
}
} // namespace

namespace sfFDN
{

DelayInterp::DelayInterp(const DelayOptions& config)
    : delayline_(static_cast<uint32_t>(config.delay), config.max_delay)
    , delay_(0)
    , int_delay_(0)
    , frac_delay_(0.0f)
    , type_(config.interp_type)
    , linear_last_out_(0.0f)
{
    this->SetDelay(config.delay);
}

void DelayInterp::Clear()
{
    delayline_.Clear();
    allpass_.Clear();
}

void DelayInterp::SetMaximumDelay(uint32_t delay)
{
    delayline_.SetMaximumDelay(delay);
}

uint32_t DelayInterp::GetMaximumDelay() const
{
    return delayline_.GetMaximumDelay();
}

void DelayInterp::SetDelay(float delay) noexcept SFFDN_NONBLOCKING
{
    delay_ = delay;
    int_delay_ = static_cast<uint32_t>(delay);
    frac_delay_ = delay - static_cast<float>(int_delay_);

    switch (type_)
    {
    case DelayInterpolationType::None:
    case DelayInterpolationType::Linear:
    {
        delayline_.SetDelay(int_delay_);
        break;
    }
    case DelayInterpolationType::Allpass:
    {
        if (frac_delay_ < 0.5f)
        {
            int_delay_ -= 1;
            frac_delay_ += 1.0f;
        }

        assert(int_delay_ >= 0);

        // To smooth out transients, when the integer value of the delay changes we run the filter with the last
        // output.
        const bool update_allpass = delayline_.GetDelay() != int_delay_;

        delayline_.SetDelay(int_delay_);

        allpass_.SetCoefficients((1.0f - frac_delay_) / (1.0f + frac_delay_));
        if (update_allpass)
        {
            allpass_.Tick(delayline_.LastOut());
        }
        break;
    }
    case DelayInterpolationType::Lagrange:
    {
        if (frac_delay_ < 1.f)
        {
            int_delay_ -= 1;
            frac_delay_ += 1.0f;
        }
        delayline_.SetDelay(int_delay_);
        const auto coeffs = GetLagrangeCoefficients<kLagrangeOrder>(frac_delay_);
        std::ranges::copy(coeffs, lagrange_coeffs_.begin());
        break;
    }
    default:
        assert(false);
    }
}

float DelayInterp::GetDelay() const
{
    return delay_;
}

float DelayInterp::Tick(float input) noexcept SFFDN_NONBLOCKING
{
    if (type_ == DelayInterpolationType::None)
    {
        return delayline_.Tick(input);
    }

    if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.Tick(input);
        const float a = delayline_.TapOut(int_delay_);
        const float b = delayline_.TapOut(int_delay_ + 1);
        return a + (b - a) * frac_delay_;
    }

    if (type_ == DelayInterpolationType::Allpass)
    {
        const float out = delayline_.Tick(input);
        return allpass_.Tick(out);
    }

    if (type_ == DelayInterpolationType::Lagrange)
    {
        // const float out = delayline_.Tick(input);
        // return lagrange_filter_.Tick(out);
        delayline_.Tick(input);
        const float xm1 = delayline_.TapOut(int_delay_);
        const float x0 = delayline_.TapOut(int_delay_ + 1);
        const float x1 = delayline_.TapOut(int_delay_ + 2);
        const float x2 = delayline_.TapOut(int_delay_ + 3);
        return xm1 * lagrange_coeffs_[0] + x0 * lagrange_coeffs_[1] + x1 * lagrange_coeffs_[2] +
               x2 * lagrange_coeffs_[3];
    }

    assert(false);
    return 0.0f;
}

void DelayInterp::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    if (type_ == DelayInterpolationType::None)
    {
        delayline_.Process(input, output);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.Process(input, output);
        std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        auto out_span = output.GetChannelSpan(0);
        out_span[0] = out_span[0] * coeffs[0] + linear_last_out_ * coeffs[1];
        linear_last_out_ = out_span[0];
        for (uint32_t n = 1; n < out_span.size(); ++n)
        {
            const float tmp = out_span[n];
            out_span[n] = out_span[n] * coeffs[0] + linear_last_out_ * coeffs[1];
            linear_last_out_ = tmp;
        }
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        delayline_.Process(input, output);
        allpass_.Process(output, output);
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        const auto input_span = input.GetChannelSpan(0);
        auto output_span = output.GetChannelSpan(0);
        const size_t required_history = input_span.size() + int_delay_ + kLagrangeOrder;

        if (!delayline_.CanAddNextInputs(input_span.size()) || required_history > delayline_.GetMaximumDelay())
        {
            for (size_t i = 0; i < input_span.size(); ++i)
            {
                output_span[i] = Tick(input_span[i]);
            }
            return;
        }

        if (!delayline_.AddNextInputs(input_span))
        {
            assert(false);
            return;
        }
        std::ranges::fill(output_span, 0.f);
        std::array<uint32_t, kLagrangeTapCount> taps = {int_delay_, int_delay_ + 1, int_delay_ + 2, int_delay_ + 3};
        delayline_.GetNextOutputsAt(taps, output_span, lagrange_coeffs_);
        delayline_.AdvanceRead(input_span.size());
    }
}

bool DelayInterp::AddNextInputs(std::span<const float> input) noexcept SFFDN_NONBLOCKING
{
    return delayline_.AddNextInputs(input);
}

void DelayInterp::GetNextOutputs(std::span<float> output) noexcept SFFDN_NONBLOCKING
{
    if (type_ == DelayInterpolationType::None)
    {
        delayline_.GetNextOutputs(output);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.GetNextOutputs(output);
        std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        output[0] = output[0] * coeffs[0] + linear_last_out_ * coeffs[1];
        linear_last_out_ = output[0];
        for (uint32_t n = 1; n < output.size(); ++n)
        {
            const float tmp = output[n];
            output[n] = output[n] * coeffs[0] + linear_last_out_ * coeffs[1];
            linear_last_out_ = tmp;
        }
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        delayline_.GetNextOutputs(output);
        AudioBuffer output_buffer(output);
        allpass_.Process(output_buffer, output_buffer);
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        std::ranges::fill(output, 0.f);
        std::array<uint32_t, kLagrangeTapCount> taps = {int_delay_, int_delay_ + 1, int_delay_ + 2, int_delay_ + 3};
        delayline_.GetNextOutputsAt(taps, output, lagrange_coeffs_);
        delayline_.AdvanceRead(output.size());
    }
}

std::unique_ptr<AudioProcessor> DelayInterp::Clone() const
{
    return std::make_unique<DelayInterp>(*this);
}

} // namespace sfFDN