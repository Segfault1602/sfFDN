#include "sffdn/filter.h"
#include "sffdn/filter_design.h"

#include "pch.h"

namespace
{
constexpr float TWO_PI = std::numbers::pi_v<float> * 2;
}

namespace sfFDN
{

OnePoleFilter::OnePoleFilter()
    : gain_(1.0f)
    , b0_(1.0f)
    , a1_(0.0f)
    , state_{0.0f, 0.0f}
{
}

void OnePoleFilter::SetT60s(float dc, float ny, uint32_t delay, float sample_rate)
{
    GetOnePoleAbsorption(dc, ny, sample_rate, delay, b0_, a1_);
}

void OnePoleFilter::SetPole(float pole)
{
    // https://ccrma.stanford.edu/~jos/fp/One_Pole.html
    // If the filter has a pole at z = -a, then a_[1] will be -pole;
    assert(pole <= 1.f && pole >= -1.f);

    // Set the b value to 1 - |a| to get a peak gain of 1.
    b0_ = 1.f - std::abs(pole);
    a1_ = -pole;
}

void OnePoleFilter::SetCoefficients(float b0, float a1)
{
    b0_ = b0;
    a1_ = a1;
}

void OnePoleFilter::SetDecayFilter(float decayDb, float timeMs, float samplerate)
{
    assert(decayDb < 0.f);
    const float lambda = std::log(std::pow(10.f, (decayDb / 20.f)));
    const float pole = std::exp(lambda / (timeMs / 1000.f) / samplerate);
    SetPole(pole);
}

void OnePoleFilter::SetLowpass(float cutoff)
{
    assert(cutoff >= 0.f && cutoff <= 1.f);
    const float wc = TWO_PI * cutoff;
    const float y = 1 - std::cos(wc);
    const float p = -y + std::sqrt(y * y + 2 * y);
    SetPole(1 - p);
}

float OnePoleFilter::Tick(float in)
{
    state_[0] = gain_ * in * b0_ - state_[1] * a1_;
    state_[1] = state_[0];
    return state_[0];
}

void OnePoleFilter::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // OnePoleFilter only supports single channel input/output

    auto input_span = input.GetChannelSpan(0);
    auto output_span = output.GetChannelSpan(0);
    for (auto i = 0; i < input_span.size(); ++i)
    {
        output_span[i] = Tick(input_span[i]);
    }
}
uint32_t OnePoleFilter::InputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel input
}

uint32_t OnePoleFilter::OutputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel output
}

void OnePoleFilter::Clear()
{
    std::ranges::fill(state_, 0.f);
}

std::unique_ptr<AudioProcessor> OnePoleFilter::Clone() const
{
    auto clone = std::make_unique<OnePoleFilter>();
    clone->SetCoefficients(b0_, a1_);
    clone->gain_ = gain_;
    return clone;
}

AllpassFilter::AllpassFilter()
    : coeff_(0.0f)
    , last_in_(0.0f)
    , last_out_(0.0f)
{
}

void AllpassFilter::SetCoefficients(float coeff)
{
    coeff_ = coeff;
}

float AllpassFilter::Tick(float in)
{
    last_out_ = -coeff_ * last_out_ + last_in_ + (coeff_ * in);
    last_in_ = in;
    return last_out_;
}

void AllpassFilter::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // OnePoleFilter only supports single channel input/output

    auto input_span = input.GetChannelSpan(0);
    auto output_span = output.GetChannelSpan(0);
    for (auto i = 0; i < input_span.size(); ++i)
    {
        last_out_ = coeff_ * (input_span[i] - last_out_) + last_in_;
        last_in_ = input_span[i];
        output_span[i] = last_out_;
    }
}
uint32_t AllpassFilter::InputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel input
}

uint32_t AllpassFilter::OutputChannelCount() const
{
    return 1; // OnePoleFilter only supports single channel output
}

void AllpassFilter::Clear()
{
    last_in_ = 0.f;
    last_out_ = 0.f;
}

std::unique_ptr<AudioProcessor> AllpassFilter::Clone() const
{
    auto clone = std::make_unique<AllpassFilter>();
    clone->SetCoefficients(coeff_);
    return clone;
}

} // namespace sfFDN