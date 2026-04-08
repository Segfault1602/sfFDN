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
    std::array<float, N + 1> coeffs;
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

DelayInterp::DelayInterp(float delay, uint32_t max_delay, DelayInterpolationType type)
    : delayline_(static_cast<uint32_t>(delay + 1), max_delay)
    , delay_(0)
    , int_delay_(0)
    , frac_delay_(0.0f)
    , type_(type)
{
    this->SetDelay(delay);
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

void DelayInterp::SetDelay(float delay)
{
    delay_ = delay;
    int_delay_ = static_cast<uint32_t>(delay);
    frac_delay_ = delay - static_cast<float>(int_delay_);

    if (type_ == DelayInterpolationType::None)
    {
        delayline_.SetDelay(int_delay_);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.SetDelay(int_delay_);
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        if (frac_delay_ < 0.5f)
        {
            int_delay_ -= 1;
            frac_delay_ += 1.0f;
        }

        assert(int_delay_ >= 0);

        // To smooth out transients, when the integer value of the delay changes we run the filter with the last output.
        const bool update_allpass = delayline_.GetDelay() != int_delay_;

        delayline_.SetDelay(int_delay_);

        allpass_.SetCoefficients((1.0f - frac_delay_) / (1.0f + frac_delay_));
        if (update_allpass)
        {
            allpass_.Tick(delayline_.LastOut());
        }
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        if (frac_delay_ < 1.f)
        {
            int_delay_ -= 1;
            frac_delay_ += 1.0f;
        }
        delayline_.SetDelay(int_delay_);
        std::array<float, 4> coeffs = GetLagrangeCoefficients<3>(frac_delay_);
        lagrange_coeffs_.resize(coeffs.size());
        std::ranges::copy(coeffs, lagrange_coeffs_.begin());
        lagrange_filter_.SetCoefficients(coeffs);
    }
}

float DelayInterp::GetDelay() const
{
    return delay_;
}

float DelayInterp::Tick(float input)
{
    if (type_ == DelayInterpolationType::None)
    {
        return delayline_.Tick(input);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.Tick(input);
        const float a = delayline_.TapOut(int_delay_);
        const float b = delayline_.TapOut(int_delay_ + 1);
        return a + (b - a) * frac_delay_;
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        const float out = delayline_.Tick(input);
        return allpass_.Tick(out);
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        const float out = delayline_.Tick(input);
        return lagrange_filter_.Tick(out);
    }

    assert(false);
    return 0.0f;
}

void DelayInterp::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    auto in_span = input.GetChannelSpan(0);
    auto out_span = output.GetChannelSpan(0);

    if (type_ == DelayInterpolationType::None)
    {
        delayline_.Process(input, output);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        if (delayline_.AddNextInputs(in_span))
        {
            std::array<uint32_t, 2> taps = {int_delay_, int_delay_ + 1};
            std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
            delayline_.GetNextOutputsAt(taps, out_span, coeffs);
        }
        else
        {
            for (uint32_t n = 0; n < input.SampleCount(); ++n)
            {
                out_span[n] = this->Tick(in_span[n]);
            }
        }
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        delayline_.Process(input, output);
        allpass_.Process(output, output);
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        delayline_.Process(input, output);
        lagrange_filter_.Process(output, output);
    }
}

bool DelayInterp::AddNextInputs(std::span<const float> input)
{
    return delayline_.AddNextInputs(input);
}

void DelayInterp::GetNextOutputs(std::span<float> output)
{
    if (type_ == DelayInterpolationType::None)
    {
        delayline_.GetNextOutputs(output);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        std::array<uint32_t, 2> taps = {int_delay_, int_delay_ + 1};
        std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        delayline_.GetNextOutputsAt(taps, output, coeffs);
    }
    else if (type_ == DelayInterpolationType::Allpass)
    {
        delayline_.GetNextOutputs(output);
        AudioBuffer output_buffer(output);
        allpass_.Process(output_buffer, output_buffer);
    }
    else if (type_ == DelayInterpolationType::Lagrange)
    {
        delayline_.GetNextOutputs(output);
        AudioBuffer output_buffer(output);
        lagrange_filter_.Process(output_buffer, output_buffer);
    }
}

NLOHMANN_JSON_SERIALIZE_ENUM(DelayInterpolationType, {
                                                         {DelayInterpolationType::None, "None"},
                                                         {DelayInterpolationType::Linear, "Linear"},
                                                         {DelayInterpolationType::Allpass, "Allpass"},
                                                         {DelayInterpolationType::Lagrange, "Lagrange"},
                                                     })

nlohmann::json DelayInterp::ToJson() const
{
    nlohmann::json j;
    j["type"] = "DelayInterp";
    j["delay"] = delay_;
    j["max_delay"] = delayline_.GetMaximumDelay();

    j["interpolation"] = type_;

    return j;
}

DelayInterp DelayInterp::FromJson(const nlohmann::json& j)
{
    ThrowIfNotType(j, "DelayInterp");
    float delay = j.at("delay").get<float>();
    uint32_t max_delay = j.at("max_delay").get<uint32_t>();
    DelayInterpolationType type = j.at("interpolation").get<DelayInterpolationType>();

    DelayInterp delay_interp(delay, max_delay, type);

    return delay_interp;
}

} // namespace sfFDN