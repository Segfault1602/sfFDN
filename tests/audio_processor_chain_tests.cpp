#include "doctest.h"

#include <cstddef>
#include <memory>
#include <print>
#include <vector>

#include "sffdn/sffdn.h"

TEST_CASE("AudioProcessorChain")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 4;

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
    CHECK(chain.OutputChannelCount() == N);

    chain.AddProcessor(std::move(output_gains));

    CHECK(chain.InputChannelCount() == 1);
    CHECK(chain.OutputChannelCount() == 1);

    std::vector<float> input(kBlockSize, 1.f);
    std::vector<float> output(kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output.data());
    chain.Process(input_buffer, output_buffer);

    for (size_t i = 0; i < output.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(5.f));
    }
}