#include "schroeder_allpass.h"

#include <cassert>
#include <mdspan>

#include <arm_neon.h>

namespace fdn
{
SchroederAllpass::SchroederAllpass(size_t delay, float g)
    : delay_(delay, (delay < 32) ? 32 : delay + 1)
    , g_(g)
{
}

void SchroederAllpass::SetDelay(size_t delay)
{
    delay_.SetMaximumDelay(delay + 1);
    delay_.SetDelay(delay);
}

void SchroederAllpass::SetG(float g)
{
    g_ = g;
}

float SchroederAllpass::Tick(float input)
{
    float out = delay_.NextOut();
    float v_n = input - g_ * out;
    delay_.Tick(v_n);
    return g_ * v_n + out;
}

void SchroederAllpass::ProcessBlock(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == out.size());

    size_t unroll_size = in.size() & ~3;

    if (delay_.GetDelay() < 4)
    {
        unroll_size = 0;
    }

    for (size_t i = 0; i < unroll_size; i += 4)
    {
        float del_out[4];
        delay_.GetNextOutputs(del_out);

        float v_n0 = in[i] - g_ * del_out[0];
        float v_n1 = in[i + 1] - g_ * del_out[1];
        float v_n2 = in[i + 2] - g_ * del_out[2];
        float v_n3 = in[i + 3] - g_ * del_out[3];

        delay_.AddNextInputs({{v_n0, v_n1, v_n2, v_n3}});

        out[i] = g_ * v_n0 + del_out[0];
        out[i + 1] = g_ * v_n1 + del_out[1];
        out[i + 2] = g_ * v_n2 + del_out[2];
        out[i + 3] = g_ * v_n3 + del_out[3];
    }

    for (size_t i = unroll_size; i < in.size(); ++i)
    {
        out[i] = Tick(in[i]);
    }
}

SchroederAllpassSection::SchroederAllpassSection(size_t N)
    : stage_(N)
{
    allpasses_.reserve(N);
    for (size_t i = 0; i < N; i++)
    {
        allpasses_.emplace_back(1, 0.0f);
    }
}

void SchroederAllpassSection::SetDelays(std::span<size_t> delays)
{
    for (size_t i = 0; i < delays.size(); i++)
    {
        allpasses_[i].SetDelay(delays[i]);
    }
}

void SchroederAllpassSection::SetGains(std::span<float> gains)
{
    assert(gains.size() == allpasses_.size());
    for (size_t i = 0; i < gains.size(); i++)
    {
        allpasses_[i].SetG(gains[i]);
    }
}

size_t SchroederAllpassSection::InputChannelCount() const
{
    return allpasses_.size();
}

size_t SchroederAllpassSection::OutputChannelCount() const
{
    return allpasses_.size();
}

void SchroederAllpassSection::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == stage_);

    for (size_t i = 0; i < allpasses_.size(); ++i)
    {
        auto input_span = input.GetChannelSpan(i);
        auto output_span = output.GetChannelSpan(i);
        allpasses_[i].ProcessBlock(input_span, output_span);
    }
}

} // namespace fdn