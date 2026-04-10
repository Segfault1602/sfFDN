#include "sffdn/filter.h"

#include "json_helper.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <utility>

namespace
{
float ComputeSample(float x, const sfFDN::FilterCoefficients& coeffs, sfFDN::CascadedBiquads::IIRState& state)
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

CascadedBiquads::CascadedBiquads(const CascadedBiquadsConfig& config)
    : stage_(config.coeffs.size())
{
    SetCoefficients(config.coeffs);
}

void CascadedBiquads::SetCoefficients(std::span<const FilterCoefficients> coeffs)
{
    coeffs_.clear();
    coeffs_.resize(coeffs.size());

    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        coeffs_[i] = coeffs[i].Normalize();
    }

    states_.resize(coeffs.size(), {.s0 = 0.0f, .s1 = 0.0f});
    stage_ = static_cast<uint32_t>(coeffs.size());
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
        const auto& coeffs = coeffs_[i];
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
            const FilterCoefficients& coeffs = coeffs_[stage];
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
            const FilterCoefficients& coeffs = coeffs_[stage];
            IIRState& state = states_[stage];
            s = ComputeSample(s, coeffs, state);
        }

        out[sample] = s;
    }
}

uint32_t CascadedBiquads::InputChannelCount() const
{
    return 1;
}

uint32_t CascadedBiquads::OutputChannelCount() const
{
    return 1;
}

std::unique_ptr<AudioProcessor> CascadedBiquads::Clone() const
{
    auto clone = std::make_unique<CascadedBiquads>(*this);
    return clone;
}

nlohmann::json CascadedBiquads::ToJson() const
{
    nlohmann::json j;
    j["type"] = "CascadedBiquads";
    j["coefficients"] = nlohmann::json::array();
    for (const auto& coeffs : coeffs_)
    {
        std::array<float, 6> coeff_array = {coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a0, coeffs.a1, coeffs.a2};
        j["coefficients"].push_back(coeff_array);
    }
    return j;
}

std::unique_ptr<CascadedBiquads> CascadedBiquads::FromJson(const nlohmann::json& j)
{
    ThrowIfNotType(j, "CascadedBiquads");
    ThrowIfDoesNotContainKey(j, "coefficients");

    auto coeffs_json = j["coefficients"];
    if (!coeffs_json.is_array())
    {
        throw std::invalid_argument("Coefficients must be an array.");
    }

    std::vector<FilterCoefficients> coeffs;
    for (const auto& coeffs_entry : coeffs_json)
    {
        if (!coeffs_entry.is_array() || coeffs_entry.size() != 6)
        {
            throw std::invalid_argument("Each coefficient entry must be an array of 6 floats.");
        }

        FilterCoefficients fc;
        fc.b0 = coeffs_entry[0].get<float>();
        fc.b1 = coeffs_entry[1].get<float>();
        fc.b2 = coeffs_entry[2].get<float>();
        fc.a0 = coeffs_entry[3].get<float>();
        fc.a1 = coeffs_entry[4].get<float>();
        fc.a2 = coeffs_entry[5].get<float>();

        coeffs.push_back(fc);
    }

    auto filter = std::make_unique<CascadedBiquads>();
    filter->SetCoefficients(coeffs);
    return filter;
}

} // namespace sfFDN