#include "sffdn/delay_interp.h"

#include "sffdn/audio_buffer.h"

#include "json_helper.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <ranges>

namespace
{
/// Smallest delay the allpass structure can represent: the fractional part is kept in [0.5, 1.5).
constexpr float kMinimumAllpassDelay = 0.5f;

/// Smallest delay the 3rd order Lagrange structure can represent: the fractional part is kept in [1, 2).
constexpr float kMinimumLagrangeDelay = 1.0f;

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
    , delay_(-1.0f) // never a valid delay, so the SetDelay() below always runs
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
    next_out_ = 0.f;
    has_next_out_ = false;
    linear_last_out_ = 0.f;
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
    if (delay == delay_)
    {
        // Nothing downstream of this function depends on anything but the delay, so recomputing it would produce the
        // exact same state. `has_next_out_` is deliberately left alone: a cached output computed for this same delay
        // is still valid.
        return;
    }

    delay_ = delay;
    has_next_out_ = false;

    switch (type_)
    {
    case DelayInterpolationType::None:
    case DelayInterpolationType::Linear:
    {
        const auto new_int_delay = static_cast<uint32_t>(delay);
        frac_delay_ = delay - static_cast<float>(new_int_delay);

        const bool tap_moved = delayline_.GetDelay() != new_int_delay;
        int_delay_ = new_int_delay;
        delayline_.SetDelay(int_delay_);

        if (tap_moved)
        {
            // Same reasoning as the allpass branch below: the block path remembers the previous raw output of the
            // delay line, which belongs to the old tap once the integer part of the delay changes. While the tap
            // stays put the stored sample is already the right one, so the read is skipped.
            linear_last_out_ = delayline_.TapOut(int_delay_);
        }
        break;
    }
    case DelayInterpolationType::Allpass:
    {
        // The fractional part is kept in [0.5, 1.5) so that the allpass coefficient stays in [-1/3, 1/3]. Delays
        // below kMinimumAllpassDelay cannot be represented by this structure and are clamped.
        const float clamped = std::max(delay, kMinimumAllpassDelay);
        float int_part = std::floor(clamped);
        float frac = clamped - int_part;
        if (frac < 0.5f)
        {
            int_part -= 1.0f;
            frac += 1.0f;
        }

        assert(int_part >= 0.0f);
        const auto new_int_delay = static_cast<uint32_t>(int_part);
        frac_delay_ = frac;
        const float coeff = (1.0f - frac_delay_) / (1.0f + frac_delay_);

        const bool tap_moved = delayline_.GetDelay() != new_int_delay;
        int_delay_ = new_int_delay;
        delayline_.SetDelay(int_delay_);

        if (tap_moved)
        {
            // The allpass state must be seeded with the previous sample of the *new* tap. Taps are counted from the
            // write pointer and no sample has been written since the last output, so TapOut(int_delay_) is exactly
            // the sample that precedes the one the next Tick() will read. Seeding it with the *old* tap instead, as
            // feeding Delay::LastOut() back through Tick() does, injects a step into the output every time the
            // integer part of the delay changes.
            allpass_.WarpState(delayline_.TapOut(int_delay_), coeff);
        }
        else
        {
            // The tap did not move, so the stored previous input is already the right sample.
            allpass_.SetCoefficients(coeff);
        }
        break;
    }
    case DelayInterpolationType::Lagrange:
    {
        // The four taps straddle the fractional delay, so the fractional part is kept in [1, 2). Delays below
        // kMinimumLagrangeDelay cannot be represented and are clamped.
        const float clamped = std::max(delay, kMinimumLagrangeDelay);
        const float int_part = std::floor(clamped) - 1.0f;

        assert(int_part >= 0.0f);
        int_delay_ = static_cast<uint32_t>(int_part);
        frac_delay_ = clamped - std::floor(clamped) + 1.0f;

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
    has_next_out_ = false;

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

float DelayInterp::NextOut() noexcept SFFDN_NONBLOCKING
{
    if (has_next_out_)
    {
        return next_out_;
    }

    // The taps below are the ones used by Tick(), shifted down by one, because Delay::TapOut() is counted from the
    // write pointer and Tick() reads after writing the current input sample.
    switch (type_)
    {
    case DelayInterpolationType::None:
    {
        next_out_ = delayline_.NextOut();
        break;
    }
    case DelayInterpolationType::Linear:
    {
        assert(int_delay_ >= 1);
        const float a = delayline_.TapOut(int_delay_ - 1);
        const float b = delayline_.TapOut(int_delay_);
        next_out_ = a + (b - a) * frac_delay_;
        break;
    }
    case DelayInterpolationType::Allpass:
    {
        next_out_ = allpass_.Tick(delayline_.NextOut());
        break;
    }
    case DelayInterpolationType::Lagrange:
    {
        assert(int_delay_ >= 1);
        const float xm1 = delayline_.TapOut(int_delay_ - 1);
        const float x0 = delayline_.TapOut(int_delay_);
        const float x1 = delayline_.TapOut(int_delay_ + 1);
        const float x2 = delayline_.TapOut(int_delay_ + 2);
        next_out_ =
            xm1 * lagrange_coeffs_[0] + x0 * lagrange_coeffs_[1] + x1 * lagrange_coeffs_[2] + x2 * lagrange_coeffs_[3];
        break;
    }
    default:
        assert(false);
        next_out_ = 0.0f;
        break;
    }

    has_next_out_ = true;
    return next_out_;
}

void DelayInterp::Advance(float input) noexcept SFFDN_NONBLOCKING
{
    delayline_.Tick(input);
    has_next_out_ = false;
}

float DelayInterp::TapOut(uint32_t tap) const noexcept SFFDN_NONBLOCKING
{
    return delayline_.TapOut(tap);
}

void DelayInterp::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    has_next_out_ = false;

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
        const std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        auto out_span = output.GetChannelSpan(0);
        for (uint32_t n = 0; n < out_span.size(); ++n)
        {
            // linear_last_out_ holds the *raw* delay line output of the previous sample, so it has to be saved
            // before the interpolated value overwrites it.
            const float raw = out_span[n];
            out_span[n] = (raw * coeffs[0]) + (linear_last_out_ * coeffs[1]);
            linear_last_out_ = raw;
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
    has_next_out_ = false;
    return delayline_.AddNextInputs(input);
}

void DelayInterp::GetNextOutputs(std::span<float> output) noexcept SFFDN_NONBLOCKING
{
    has_next_out_ = false;

    if (type_ == DelayInterpolationType::None)
    {
        delayline_.GetNextOutputs(output);
    }
    else if (type_ == DelayInterpolationType::Linear)
    {
        delayline_.GetNextOutputs(output);
        const std::array<float, 2> coeffs = {1.0f - frac_delay_, frac_delay_};
        for (uint32_t n = 0; n < output.size(); ++n)
        {
            const float raw = output[n];
            output[n] = (raw * coeffs[0]) + (linear_last_out_ * coeffs[1]);
            linear_last_out_ = raw;
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
        // Unlike Process(), this is the read-before-write half of the block API: the caller reads a block out and
        // only then writes the next block in. Delay::GetNextOutputsAt() anchors its taps on the write pointer and
        // assumes the block has already been written, so every tap has to be pulled back by one block to compensate.
        // That is only possible when the delay covers a whole block, which is the same requirement the integer path
        // has.
        const auto block_size = static_cast<uint32_t>(output.size());
        if (int_delay_ < block_size || int_delay_ + kLagrangeOrder > delayline_.GetMaximumDelay())
        {
            assert(false);
            std::ranges::fill(output, 0.f);
            return;
        }

        std::ranges::fill(output, 0.f);
        const uint32_t first_tap = int_delay_ - block_size;
        std::array<uint32_t, kLagrangeTapCount> taps = {first_tap, first_tap + 1, first_tap + 2, first_tap + 3};
        delayline_.GetNextOutputsAt(taps, output, lagrange_coeffs_);
        delayline_.AdvanceRead(output.size());
    }
}

std::unique_ptr<AudioProcessor> DelayInterp::Clone() const
{
    return std::make_unique<DelayInterp>(*this);
}

} // namespace sfFDN