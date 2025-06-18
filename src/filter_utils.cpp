#include "filter_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
std::vector<float> ComputeRMS(const std::vector<float>& samples, size_t win_size)
{
    std::vector<float> rms(samples.size(), 0.f);
    size_t half_win_size = win_size / 2;

    for (size_t i = 0; i < samples.size(); ++i)
    {
        size_t start = (i < half_win_size) ? 0 : i - half_win_size;
        size_t end = std::min(i + half_win_size, samples.size() - 1);

        float sum_squares = 0.f;
        for (size_t j = start; j <= end; ++j)
        {
            sum_squares += samples[j] * samples[j];
        }
        rms[i] = 20 * std::log10(std::sqrt(sum_squares / (end - start + 1)));
    }

    return rms;
}
} // namespace

namespace fdn
{
std::vector<float> GetImpulseResponse(Filter* filter, size_t block_size)
{
    if (!filter)
    {
        return {};
    }

    constexpr size_t kRMSWinSize = 128;
    constexpr size_t kBlockSize = 512;
    constexpr size_t kMaxSamples = 32768;

    std::vector<float> impulse;
    impulse.reserve(kMaxSamples);
    impulse.push_back(filter->Tick(1.f));

    std::vector<float> level;
    level.reserve(kMaxSamples);

    OnePoleFilter one_pole_filter;
    one_pole_filter.SetPole(0.99f);

    level.push_back(one_pole_filter.Tick(impulse.back() * impulse.back()));

    for (size_t i = 0; i < kMaxSamples; ++i)
    {
        impulse.push_back(filter->Tick(0.f));
        level.push_back(one_pole_filter.Tick(impulse.back() * impulse.back()));

        if (level.back() < 5e-6f) // Threshold to stop the impulse response
        {
            break;
        }
    }

    return impulse;
}
} // namespace fdn