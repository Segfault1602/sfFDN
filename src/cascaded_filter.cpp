#include "sffdn/filter.h"

#include <cassert>
#include <iostream>
#include <span>

namespace sfFDN
{

CascadedBiquads::CascadedBiquads()
    : stage_(0)
{
}

CascadedBiquads::~CascadedBiquads()
{
}

void CascadedBiquads::SetCoefficients(uint32_t num_stage, std::span<const float> coeffs)
{
    coeffs_.clear();
    coeffs_.resize(num_stage);

    if (coeffs.size() == num_stage * 5)
    {
        for (uint32_t i = 0; i < num_stage; ++i)
        {
            coeffs_[i].b0 = coeffs[i * 5 + 0];
            coeffs_[i].b1 = coeffs[i * 5 + 1];
            coeffs_[i].b2 = coeffs[i * 5 + 2];
            coeffs_[i].a1 = coeffs[i * 5 + 3];
            coeffs_[i].a2 = coeffs[i * 5 + 4];
        }
    }
    else
    {
        assert(coeffs.size() == num_stage * 6);
        for (uint32_t i = 0; i < num_stage; ++i)
        {
            const float a0 = coeffs[i * 6 + 3];

            coeffs_[i].b0 = coeffs[i * 6 + 0] / a0;
            coeffs_[i].b1 = coeffs[i * 6 + 1] / a0;
            coeffs_[i].b2 = coeffs[i * 6 + 2] / a0;
            coeffs_[i].a1 = coeffs[i * 6 + 4] / a0;
            coeffs_[i].a2 = coeffs[i * 6 + 5] / a0;
        }
    }

    states_.resize(num_stage, {0});
    stage_ = num_stage;
}

void CascadedBiquads::Clear()
{
    states_.clear();
    states_.resize(stage_, {0});
}

float CascadedBiquads::Tick(float in)
{
    float out = in;
    for (uint32_t i = 0; i < stage_; ++i)
    {
        IIRCoeffs& coeffs = coeffs_[i];
        IIRState& state = states_[i];

        out = coeffs.b0 * out + state.s0;
        state.s0 = coeffs.b1 * out + state.s1 - coeffs.a1 * out;
        state.s1 = coeffs.b2 * out - coeffs.a2 * out;
    }
    return out;
}

void CascadedBiquads::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == 1);

    auto in = input.GetChannelSpan(0);
    auto out = output.GetChannelSpan(0);

    size_t sample = 0;
    const size_t kSize = in.size();
    const size_t unroll_size = kSize & ~3;
    while (sample < unroll_size)
    {
        size_t stage = 0;
        float in1 = in[sample];
        float in2 = in[sample + 1];
        float in3 = in[sample + 2];
        float in4 = in[sample + 3];

        float out1 = 0;
        float out2 = 0;
        float out3 = 0;
        float out4 = 0;
        while (stage < stage_)
        {
            IIRCoeffs coeffs = coeffs_[stage];
            float s0 = states_[stage].s0;
            float s1 = states_[stage].s1;

#define COMPUTE_SAMPLE(x, y)                                                                                           \
    y = coeffs.b0 * x + s0;                                                                                            \
    s0 = coeffs.b1 * x + s1;                                                                                           \
    s0 -= coeffs.a1 * y;                                                                                               \
    s1 = coeffs.b2 * x;                                                                                                \
    s1 -= coeffs.a2 * y;

            COMPUTE_SAMPLE(in1, out1);
            COMPUTE_SAMPLE(in2, out2);
            COMPUTE_SAMPLE(in3, out3);
            COMPUTE_SAMPLE(in4, out4);

            in1 = out1;
            in2 = out2;
            in3 = out3;
            in4 = out4;

            states_[stage].s0 = s0;
            states_[stage].s1 = s1;

            ++stage;
        }

        out[sample] = out1;
        out[sample + 1] = out2;
        out[sample + 2] = out3;
        out[sample + 3] = out4;
        sample += 4;
    }

    while (sample < kSize)
    {
        size_t stage = 0;
        float in1 = in[sample];
        float out1 = 0;
        do
        {
            IIRCoeffs coeffs = coeffs_[stage];
            IIRState* state = &states_[stage];

            out1 = coeffs.b0 * in1 + state->s0;
            state->s0 = coeffs.b1 * in1 - coeffs.a1 * out1 + state->s1;
            state->s1 = coeffs.b2 * in1 - coeffs.a2 * out1;

            in1 = out1;
            ++stage;
        } while (stage < stage_);

        out[sample] = out1;
        ++sample;
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

void CascadedBiquads::dump_coeffs()
{
    for (uint32_t i = 0; i < stage_; i++)
    {
        uint32_t offset = i * 5;
        std::cout << "[" << coeffs_[offset].b0 << ", " << coeffs_[offset].b1 << ", " << coeffs_[offset].b2 << ", "
                  << coeffs_[offset].a1 << ", " << coeffs_[offset].a2 << "]" << std::endl;
    }
}

} // namespace sfFDN