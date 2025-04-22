#include "delay_matrix.h"

#include <cassert>
#include <iostream>

namespace fdn
{
DelayMatrix::DelayMatrix(size_t N, std::span<const size_t> delays)
    : N_(N)
{
    delays_.reserve(N * N);
    for (size_t i = 0; i < N; ++i)
    {
        delays_.emplace_back(delays[i], 4096);
    }
}

void DelayMatrix::Clear()
{
    for (auto& delay : delays_)
    {
        delay.Clear();
    }
}

void DelayMatrix::SetDelays(std::span<size_t> delays)
{
    assert(delays.size() == delays_.size());
    for (size_t i = 0; i < delays.size(); i++)
    {
        size_t max_delay = (delays[i] < 2048) ? 2048 : delays[i] * 2;
        delays_[i].SetMaximumDelay(max_delay);

        delays_[i].SetDelay(delays[i]);
    }
}

void DelayMatrix::SetMatrix(MixMat mixing_matrix)
{
    assert(mixing_matrix.GetSize() == N_);
    mixing_matrix_ = mixing_matrix;
}

void DelayMatrix::Tick(std::span<const float> input, std::span<float> output)
{
    assert(input.size() == output.size());
    // Input size must be a multiple of the delay size.
    assert(input.size() % delays_.size() == 0);

    mixing_matrix_.Tick(input, output);

    const size_t delay_count = delays_.size();
    const size_t block_size = input.size() / delay_count;

    for (size_t i = 0; i < delay_count; i++)
    {
        auto input_span = std::span<const float>(input.data() + i * block_size, block_size);
        auto output_span = std::span<float>(output.data() + i * block_size, block_size);
        delays_[i].Tick(input_span, output_span);
    }
}

void DelayMatrix::DumpDelays() const
{
    for (size_t i = 0; i < delays_.size(); i++)
    {
        std::cout << delays_[i].GetDelay() << " ";
    }
    std::cout << std::endl;
}

} // namespace fdn