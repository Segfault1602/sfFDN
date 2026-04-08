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

template <DelayInterpolationType type>
DelayInterp<type>::DelayInterp(float delay, uint32_t max_delay)
    : delayline_(static_cast<uint32_t>(delay + 1), max_delay)
    , delay_(0)
    , int_delay_(0)
    , frac_delay_(0.0f)
{
    this->SetDelay(delay);
}

template <DelayInterpolationType type>
void DelayInterp<type>::Clear()
{
    delayline_.Clear();
    allpass_.Clear();
}

template <DelayInterpolationType type>
void DelayInterp<type>::SetMaximumDelay(uint32_t delay)
{
    delayline_.SetMaximumDelay(delay);
}

template <DelayInterpolationType type>
uint32_t DelayInterp<type>::GetMaximumDelay() const
{
    return delayline_.GetMaximumDelay();
}

template <DelayInterpolationType type>
void DelayInterp<type>::SetDelay(float delay)
{
    delay_ = delay;
    int_delay_ = static_cast<uint32_t>(delay);
    frac_delay_ = delay - static_cast<float>(int_delay_);

    if constexpr (type == DelayInterpolationType::None)
    {
        delayline_.SetDelay(int_delay_);
    }
    else if constexpr (type == DelayInterpolationType::Linear)
    {
        delayline_.SetDelay(int_delay_);
    }
    else if constexpr (type == DelayInterpolationType::Allpass)
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
    else if constexpr (type == DelayInterpolationType::Lagrange)
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

template <DelayInterpolationType type>
float DelayInterp<type>::GetDelay() const
{
    return delay_;
}

template <DelayInterpolationType type>
float DelayInterp<type>::Tick(float input)
{
    if constexpr (type == DelayInterpolationType::None)
    {
        return delayline_.Tick(input);
    }
    else if constexpr (type == DelayInterpolationType::Linear)
    {
        delayline_.Tick(input);
        const float a = delayline_.TapOut(int_delay_);
        const float b = delayline_.TapOut(int_delay_ + 1);
        return a + (b - a) * frac_delay_;
    }
    else if constexpr (type == DelayInterpolationType::Allpass)
    {
        const float out = delayline_.Tick(input);
        return allpass_.Tick(out);
    }
    else if constexpr (type == DelayInterpolationType::Lagrange)
    {
        const float out = delayline_.Tick(input);
        return lagrange_filter_.Tick(out);
    }

    assert(false);
    return 0.0f;
}

template <DelayInterpolationType type>
void DelayInterp<type>::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // This class only works with mono input.

    auto in_span = input.GetChannelSpan(0);
    auto out_span = output.GetChannelSpan(0);

    if constexpr (type == DelayInterpolationType::None)
    {
        delayline_.Process(input, output);
    }
    else if constexpr (type == DelayInterpolationType::Linear)
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
    else if constexpr (type == DelayInterpolationType::Allpass)
    {
        delayline_.Process(input, output);
        allpass_.Process(output, output);
    }
    else if constexpr (type == DelayInterpolationType::Lagrange)
    {
        delayline_.Process(input, output);
        lagrange_filter_.Process(output, output);
    }
}

template <DelayInterpolationType type>
nlohmann::json DelayInterp<type>::ToJson() const
{
    nlohmann::json j;
    j["type"] = "DelayInterp";
    j["delay"] = delay_;
    j["max_delay"] = delayline_.GetMaximumDelay();

    if constexpr (type == DelayInterpolationType::None)
    {
        j["interpolation"] = "None";
    }
    else if constexpr (type == DelayInterpolationType::Linear)
    {
        j["interpolation"] = "Linear";
    }
    else if constexpr (type == DelayInterpolationType::Allpass)
    {
        j["interpolation"] = "Allpass";
    }
    else if constexpr (type == DelayInterpolationType::Lagrange)
    {
        j["interpolation"] = "Lagrange";
    }

    return j;
}

template <DelayInterpolationType type>
DelayInterp<type> DelayInterp<type>::FromJson(const nlohmann::json& j)
{
    ThrowIfNotType(j, "DelayInterp");
    float delay = j.at("delay").get<float>();
    uint32_t max_delay = j.at("max_delay").get<uint32_t>();
    std::string interpolation = j.at("interpolation").get<std::string>();
    assert((interpolation == "None" && type == DelayInterpolationType::None) ||
           (interpolation == "Linear" && type == DelayInterpolationType::Linear) ||
           (interpolation == "Allpass" && type == DelayInterpolationType::Allpass) ||
           (interpolation == "Lagrange" && type == DelayInterpolationType::Lagrange));

    return DelayInterp(delay, max_delay);
}

template class DelayInterp<DelayInterpolationType::None>;
template class DelayInterp<DelayInterpolationType::Linear>;
template class DelayInterp<DelayInterpolationType::Allpass>;
template class DelayInterp<DelayInterpolationType::Lagrange>;

} // namespace sfFDN