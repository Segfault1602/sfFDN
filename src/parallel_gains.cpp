#include "parallel_gains.h"

#include <cassert>

#include "array_math.h"

namespace fdn
{
ParallelGains::ParallelGains()
{
}

void ParallelGains::SetGains(std::span<const float> gains)
{
    assert(gains.size() > 0);
    gains_.assign(gains.begin(), gains.end());
}

void ParallelGains::ProcessBlock(const std::span<const float> input, std::span<float> output)
{
    if (input.size() <= output.size())
    {
        ProcessBlockMultiplexed(input, output);
    }
    else
    {
        ProcessBlockDeMultiplexed(input, output);
    }
}

void ParallelGains::ProcessBlockMultiplexed(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() <= output.size());
    assert(output.size() == input.size() * gains_.size());

    const size_t block_size = output.size() / gains_.size();
    const size_t gain_count = gains_.size();

    float* out_ptr = output.data();

    for (size_t i = 0; i < gain_count; i++)
    {
        auto output_span = std::span<float>(output.data() + i * block_size, block_size);
        // vDSP_vsmul(input.data(), 1, &gains_[i], output_span.data(), 1, block_size);
        ArrayMath::Scale(input, gains_[i], output_span);
    }
}

void ParallelGains::ProcessBlockDeMultiplexed(const std::span<const float> input, std::span<float> output)
{
    assert(input.size() > output.size());
    assert(input.size() % gains_.size() == 0);
    assert(output.size() == input.size() / gains_.size());

    const size_t block_size = input.size() / gains_.size();
    const size_t gain_count = gains_.size();

    const float* in_ptr = input.data();

    constexpr size_t unroll_factor = 4;
    const size_t remainder = block_size % unroll_factor;

    for (size_t i = 0; i < gain_count; i++)
    {
        auto input_span = std::span<const float>(input.data() + i * block_size, block_size);
        ArrayMath::ScaleAccumulate(input_span, gains_[i], output);
        // vDSP_vsma(input_span.data(), 1, &gains_[i], output.data(), 1, output.data(), 1, block_size);
    }
}

} // namespace fdn