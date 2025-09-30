#include "sffdn/filter.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <print>
#include <ranges>
#include <span>
#include <utility>

namespace
{
float ComputeSample(float x, const sfFDN::CascadedBiquads::IIRCoeffs& coeffs, sfFDN::CascadedBiquads::IIRState& state)
{
    const float y = (coeffs.b0 * x) + state.s0;
    state.s0 = (coeffs.b1 * x) + state.s1;
    state.s0 -= coeffs.a1 * y;
    state.s1 = coeffs.b2 * x;
    state.s1 -= coeffs.a2 * y;
    return y;
}
} // namespace

namespace sfFDN
{

CascadedBiquads::CascadedBiquads()
    : stage_(0)
{
}

CascadedBiquads::CascadedBiquads(const CascadedBiquads& other)
    : stage_(other.stage_)
    , states_(other.states_)
    , coeffs_(other.coeffs_)
{
    CascadedBiquads::Clear();
}

CascadedBiquads& CascadedBiquads::operator=(const CascadedBiquads& other)
{
    if (this != &other)
    {
        stage_ = other.stage_;
        states_ = other.states_;
        coeffs_ = other.coeffs_;
    }
    Clear();
    return *this;
}

CascadedBiquads::CascadedBiquads(CascadedBiquads&& other) noexcept
    : stage_(other.stage_)
    , states_(std::move(other.states_))
    , coeffs_(std::move(other.coeffs_))
{
}

CascadedBiquads& CascadedBiquads::operator=(CascadedBiquads&& other) noexcept
{
    if (this != &other)
    {
        stage_ = other.stage_;
        states_ = std::move(other.states_);
        coeffs_ = std::move(other.coeffs_);
    }
    return *this;
}

void CascadedBiquads::SetCoefficients(uint32_t num_stage, std::span<const float> coeffs)
{
    coeffs_.clear();
    coeffs_.resize(num_stage);

    constexpr auto kNormalizedCoeffCount = 5;

    if (coeffs.size() == (num_stage * kNormalizedCoeffCount))
    {
        for (uint32_t i = 0; i < num_stage; ++i)
        {
            auto offset = i * kNormalizedCoeffCount;
            coeffs_[i].b0 = coeffs[offset + 0];
            coeffs_[i].b1 = coeffs[offset + 1];
            coeffs_[i].b2 = coeffs[offset + 2];
            coeffs_[i].a1 = coeffs[offset + 3];
            coeffs_[i].a2 = coeffs[offset + 4];
        }
    }
    else
    {
        constexpr auto kCoeffCount = 6;
        assert(coeffs.size() == num_stage * kCoeffCount);
        for (uint32_t i = 0; i < num_stage; ++i)
        {
            const auto offset = i * kCoeffCount;
            const float a0 = coeffs[offset + 3];

            coeffs_[i].b0 = coeffs[offset + 0] / a0;
            coeffs_[i].b1 = coeffs[offset + 1] / a0;
            coeffs_[i].b2 = coeffs[offset + 2] / a0;
            coeffs_[i].a1 = coeffs[offset + 4] / a0;
            coeffs_[i].a2 = coeffs[offset + 5] / a0;
        }
    }

    states_.resize(num_stage, {.s0 = 0.0f, .s1 = 0.0f});
    stage_ = num_stage;
}

void CascadedBiquads::Clear()
{
    states_.clear();
    states_.resize(stage_, {.s0 = 0.0f, .s1 = 0.0f});
}

float CascadedBiquads::Tick(float in)
{
    float out = in;
    for (uint32_t i = 0; i < stage_; ++i)
    {
        const IIRCoeffs& coeffs = coeffs_[i];
        IIRState& state = states_[i];

        out = coeffs.b0 * out + state.s0;
        state.s0 = coeffs.b1 * out + state.s1 - coeffs.a1 * out;
        state.s1 = coeffs.b2 * out - coeffs.a2 * out;
    }
    return out;
}

void CascadedBiquads::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
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

        for (auto stage = 0u; stage < stage_; ++stage)
        {
            const IIRCoeffs coeffs = coeffs_[stage];
            IIRState& state = states_[stage];

            for (auto& b : batch)
            {
                b = ComputeSample(b, coeffs, state);
            }
        }

        for (auto [out, b] : std::views::zip(out_span, batch))
        {
            out = b;
        }
    }

    for (; sample < size; ++sample)
    {
        float s = in[sample];
        for (auto stage = 0u; stage < stage_; ++stage)
        {
            const IIRCoeffs coeffs = coeffs_[stage];
            IIRState& state = states_[stage];
            s = ComputeSample(s, coeffs, state);
        }

        out[sample] = s;
    }
}

uint32_t CascadedBiquads::InputChannelCount() const
{
    return 1; // This filter processes a single input channel
}

uint32_t CascadedBiquads::OutputChannelCount() const
{
    return 1;
}

std::unique_ptr<AudioProcessor> CascadedBiquads::Clone() const
{
    auto clone = std::make_unique<CascadedBiquads>();
    clone->stage_ = stage_;
    clone->states_.resize(states_.size());
    clone->coeffs_ = coeffs_;
    return clone;
}

} // namespace sfFDN