#include "doctest.h"

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "sffdn/sffdn.h"

TEST_CASE("ParallelGainsInput")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, N> gains = {0.25f, 0.5f, 0.75f, 1.f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::Multiplexed);
    parallel_gains.SetGains(gains);

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(N * kBlockSize, 0.f);

    std::iota(input.begin(), input.end(), 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output.data());

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 0.25, 0.5, 0.75, 1,   1.25, 1.5, 1.75, 2,   2.25, 0,   0.5,  1,   1.5,
                                       2, 2.5,  3,   3.5,  4,   4.5,  0,   0.75, 1.5, 2.25, 3,   3.75, 4.5, 5.25,
                                       6, 6.75, 0,   1.f,  2.f, 3.f,  4.f, 5.f,  6.f, 7.f,  8.f, 9.f};

    CHECK(output.size() == expected_out.size());
    for (auto i = 0; i < output.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_out[i]));
    }
}

TEST_CASE("ParallelGainsOutput")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 10;
    constexpr std::array<float, N> gains = {0.5f, 0.5f, 0.5f, 0.5f};
    sfFDN::ParallelGains parallel_gains(sfFDN::ParallelGainsMode::DeMultiplexed);
    parallel_gains.SetGains(gains);

    std::vector<float> input(N * kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);

    for (auto i = 0; i < N; ++i)
    {
        for (auto j = 0; j < kBlockSize; ++j)
        {
            input[i * kBlockSize + j] = j;
        }
    }

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data());

    parallel_gains.Process(input_buffer, output_buffer);

    std::vector<float> expected_out = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    CHECK(output.size() == expected_out.size());

    for (auto i = 0; i < output.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_out[i]));
    }
}