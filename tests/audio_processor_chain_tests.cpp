#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <memory>
#include <print>
#include <vector>

#include "sffdn/sffdn.h"

TEST_CASE("AudioProcessorChain")
{
    constexpr uint32_t N = 4;
    constexpr uint32_t kBlockSize = 4;

    sfFDN::AudioProcessorChain chain(kBlockSize);

    constexpr std::array<float, N> in_gains = {3.f, 1.f, 2.f, 4.f};
    std::unique_ptr<sfFDN::ParallelGains> input_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Multiplexed);
    input_gains->SetGains(in_gains);

    constexpr std::array<float, N> out_gains = {0.5f, 0.5f, 0.5f, 0.5f};
    std::unique_ptr<sfFDN::ParallelGains> output_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::DeMultiplexed);
    output_gains->SetGains(out_gains);

    chain.AddProcessor(std::move(input_gains));
    REQUIRE(chain.OutputChannelCount() == N);

    chain.AddProcessor(std::move(output_gains));

    REQUIRE(chain.InputChannelCount() == 1);
    REQUIRE(chain.OutputChannelCount() == 1);

    std::vector<float> input(kBlockSize, 1.f);
    std::vector<float> output(kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
    chain.Process(input_buffer, output_buffer);

    for (float& i : output)
    {
        REQUIRE_THAT(i, Catch::Matchers::WithinAbs(5.f, 0.0001));
    }
}