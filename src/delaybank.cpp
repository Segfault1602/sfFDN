#include "delaybank.h"

#include <cassert>
#include <mdspan>

namespace fdn
{

DelayBank::DelayBank(unsigned long delayCount, unsigned long maxDelay)
{
    for (size_t i = 0; i < delayCount; i++)
    {
        delays_.emplace_back(1, maxDelay);
    }
}

DelayBank::DelayBank(const std::span<const float> delays, unsigned long maxDelay)
{
    for (size_t i = 0; i < delays.size(); i++)
    {
        delays_.emplace_back(delays[i], maxDelay);
    }
}

void DelayBank::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

void DelayBank::SetDelays(const std::span<const float> delays)
{
    assert(delays.size() == delays_.size());
    for (size_t i = 0; i < delays.size(); i++)
    {
        delays_[i].SetMaximumDelay(delays[i] + 512);
        delays_[i].SetDelay(delays[i]);
    }
}

void DelayBank::SetModulation(float freq, float depth)
{
    for (size_t i = 0; i < delays_.size(); i++)
    {
        delays_[i].SetMod(freq, depth);
    }
}

void DelayBank::Tick(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());

    // Input size must be a multiple of the delay size.
    assert(input.size() % delays_.size() == 0);

    const size_t delay_count = delays_.size();
    const size_t block_size = input.size() / delay_count;

    auto input_mdspan = std::mdspan(input.data(), block_size, delay_count);
    auto output_mdspan = std::mdspan(output.data(), block_size, delay_count);

    for (size_t i = 0; i < delay_count; i++)
    {
        auto input_span = std::span<const float>(input.data() + i * block_size, block_size);
        auto output_span = std::span<float>(output.data() + i * block_size, block_size);
        for (size_t j = 0; j < block_size; j++)
        {
            output_span[j] = delays_[i].Tick(input_span[j]);
        }
    }
}

void DelayBank::AddNextInputs(const std::span<const float> input)
{
    assert(input.size() % delays_.size() == 0);

    const size_t delay_count = delays_.size();
    const size_t block_size = input.size() / delay_count;

    for (size_t i = 0; i < delay_count; i++)
    {
        auto input_span = std::span<const float>(input.data() + i * block_size, block_size);
        for (size_t j = 0; j < block_size; j++)
        {
            delays_[i].AddNextInput(input_span[j]);
        }
    }
}

void DelayBank::GetNextOutputs(std::span<float> output)
{
    assert(output.size() % delays_.size() == 0);

    const size_t delay_count = delays_.size();
    const size_t block_size = output.size() / delay_count;

    for (size_t i = 0; i < delay_count; i++)
    {
        auto output_span = std::span<float>(output.data() + i * block_size, block_size);
        for (size_t j = 0; j < block_size; j++)
        {
            output_span[j] = delays_[i].GetNextOutput();
        }
    }
}

} // namespace fdn