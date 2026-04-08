#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter_design.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <ranges>
#include <span>

namespace sfFDN
{

OnePoleFilter::OnePoleFilter()
    : b0_(1.0f)
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

void OnePoleFilter::SetDecayFilter(float decay_db, float time_ms, float sample_rate)
{
    assert(decay_db < 0.f);
    const float lambda = std::log(std::pow(10.f, (decay_db / 20.f)));
    const float pole = std::exp(lambda / (time_ms / 1000.f) / sample_rate);
    SetPole(pole);
}

void OnePoleFilter::SetLowpass(float cutoff)
{
    assert(cutoff >= 0.f && cutoff <= 1.f);
    const float wc = std::numbers::pi_v<float> * 2.f * cutoff;
    const float y = 1 - std::cos(wc);
    const float p = -y + std::sqrt((y * y) + (2 * y));
    SetPole(1 - p);
}

float OnePoleFilter::Tick(float in)
{
    state_[0] = in * b0_ - state_[1] * a1_;
    state_[1] = state_[0];
    return state_[0];
}

void OnePoleFilter::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1); // OnePoleFilter only supports single channel input/output

    auto in = input.GetChannelSpan(0);
    auto out = output.GetChannelSpan(0);

    constexpr uint32_t kUnrollFactor = 8;
    const uint32_t size = in.size();
    const uint32_t unroll_size = size & ~(kUnrollFactor - 1);

    uint32_t sample = 0;
    for (; sample < unroll_size; sample += kUnrollFactor)
    {
        auto in_span = in.subspan(sample, kUnrollFactor);
        auto out_span = out.subspan(sample, kUnrollFactor);

        // Filtering in a stack array seems to be faster than in-place filtering in the output channel directly
        std::array<float, kUnrollFactor> batch{};
        std::ranges::copy(in_span, batch.begin());

        for (auto& b : batch)
        {
            b = Tick(b);
        }
        std::ranges::copy(batch.begin(), batch.end(), out_span.begin());
    }

    for (; sample < size; ++sample)
    {
        out[sample] = Tick(in[sample]);
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
    return clone;
}

nlohmann::json OnePoleFilter::ToJson() const
{
    nlohmann::json j;
    j["type"] = "OnePoleFilter";
    j["b0"] = b0_;
    j["a1"] = a1_;
    return j;
}

AllpassFilter::AllpassFilter()
    : coeff_(0.0f)
    , last_in_(0.0f)
    , last_out_(0.0f)
{
}

AllpassFilter::AllpassFilter(const AllpassFilter& other)
    : coeff_(other.coeff_)
    , last_in_(other.last_in_)
    , last_out_(other.last_out_)
{
}

AllpassFilter& AllpassFilter::operator=(const AllpassFilter& other)
{
    if (this != &other)
    {
        coeff_ = other.coeff_;
        last_in_ = other.last_in_;
        last_out_ = other.last_out_;
    }
    return *this;
}

AllpassFilter::AllpassFilter(AllpassFilter&& other) noexcept
    : coeff_(other.coeff_)
    , last_in_(other.last_in_)
    , last_out_(other.last_out_)
{
}

AllpassFilter& AllpassFilter::operator=(AllpassFilter&& other) noexcept
{
    if (this != &other)
    {
        coeff_ = other.coeff_;
        last_in_ = other.last_in_;
        last_out_ = other.last_out_;
    }
    return *this;
}

float AllpassFilter::Tick(float in)
{
    last_out_ = coeff_ * (in - last_out_) + last_in_;
    last_in_ = in;
    return last_out_;
}

void AllpassFilter::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1);

    auto in = input.GetChannelSpan(0);
    auto out = output.GetChannelSpan(0);

    constexpr uint32_t kUnrollFactor = 8;
    const uint32_t size = in.size();
    const uint32_t unroll_size = size & ~(kUnrollFactor - 1);

    uint32_t sample = 0;
    for (; sample < unroll_size; sample += kUnrollFactor)
    {
        auto in_span = in.subspan(sample, kUnrollFactor);
        auto out_span = out.subspan(sample, kUnrollFactor);

        // Filtering in a stack array seems to be faster than in-place filtering in the output channel directly
        std::array<float, kUnrollFactor> batch{};
        std::ranges::copy(in_span, batch.begin());

        for (auto& b : batch)
        {
            b = Tick(b);
        }
        std::ranges::copy(batch.begin(), batch.end(), out_span.begin());
    }

    for (; sample < size; ++sample)
    {
        out[sample] = Tick(in[sample]);
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

nlohmann::json AllpassFilter::ToJson() const
{
    nlohmann::json j;
    j["type"] = "AllpassFilter";
    j["coeff"] = coeff_;
    return j;
}

} // namespace sfFDN