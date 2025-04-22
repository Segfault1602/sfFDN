#include "schroeder_allpass.h"

#include <cassert>
#include <mdspan>

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
    float v_n = input - g_ * delay_.LastOut();
    float out = g_ * v_n + delay_.LastOut();
    delay_.Tick(v_n);
    return out;
}

void SchroederAllpass::ProcessBlock(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == out.size());

    for (size_t i = 0; i < in.size(); i++)
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

void SchroederAllpassSection::ProcessBlock(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == out.size());

    assert(in.size() % stage_ == 0);

    const size_t block_size = in.size() / stage_;

    for (size_t i = 0; i < allpasses_.size(); ++i)
    {
        auto input_span = std::span<const float>(in.data() + i * block_size, block_size);
        auto output_span = std::span<float>(out.data() + i * block_size, block_size);
        allpasses_[i].ProcessBlock(input_span, output_span);
    }
}

} // namespace fdn