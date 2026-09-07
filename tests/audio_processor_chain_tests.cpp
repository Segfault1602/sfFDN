#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <memory>
#include <print>
#include <vector>

#include "sffdn/sffdn.h"

#include "allocation_counter.h"

TEST_CASE("AudioProcessorChain processes configured gain processors", "[processor_chain]")
{
    constexpr uint32_t kSize = 4;
    constexpr uint32_t kBlockSize = 4;

    sfFDN::AudioProcessorChain chain(kBlockSize);

    constexpr std::array<float, kSize> kInputGains = {3.f, 1.f, 2.f, 4.f};
    std::unique_ptr<sfFDN::ParallelGains> input_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split);
    input_gains->SetGains(kInputGains);

    constexpr std::array<float, kSize> kOutputGains = {0.5f, 0.5f, 0.5f, 0.5f};
    std::unique_ptr<sfFDN::ParallelGains> output_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge);
    output_gains->SetGains(kOutputGains);

    chain.AddProcessor(std::move(input_gains));
    REQUIRE(chain.OutputChannelCount() == kSize);

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

TEST_CASE("AudioProcessorChain preserves output when empty", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 4;

    std::array<float, kBlockSize> input = {1.f, 2.f, 3.f, 4.f};
    std::array<float, kBlockSize> output = {-1.f, -1.f, -1.f, -1.f};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    SECTION("empty chain leaves output unchanged")
    {
        sfFDN::AudioProcessorChain chain(kBlockSize);
        chain.Process(input_buffer, output_buffer);
        REQUIRE(output == std::array{-1.f, -1.f, -1.f, -1.f});
    }
}

TEST_CASE("AudioProcessorChain manages, processes, and clones processors", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 4;
    sfFDN::AudioProcessorChain chain(kBlockSize);

    auto split = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split);
    split->SetGains(std::array{2.f, 3.f});
    auto parallel = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    parallel->SetGains(std::array{5.f, 7.f});
    auto merge = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge);
    merge->SetGains(std::array{11.f, 13.f});

    REQUIRE(chain.AddProcessor(std::move(split)));
    REQUIRE(chain.AddProcessor(std::move(parallel)));
    REQUIRE(chain.AddProcessor(std::move(merge)));
    REQUIRE(chain.GetProcessorCount() == 3);
    REQUIRE(chain.GetProcessor(0) != nullptr);
    REQUIRE(chain.GetProcessor(2) != nullptr);
    REQUIRE(chain.GetProcessor(3) == nullptr);

    std::array<float, 7> input = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::array<float, 7> output{};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    chain.Process(input_buffer, output_buffer);

    for (size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE(output[i] == Catch::Approx(input[i] * 383.f));
    }

    chain.Clear();
    auto clone = chain.Clone();
    std::array<float, 7> clone_output{};
    sfFDN::AudioBuffer clone_output_buffer(clone_output);
    clone->Process(input_buffer, clone_output_buffer);
    REQUIRE(clone_output == output);

    {
        sfFDNTest::ScopedAllocationCounter const allocation_counter;
        chain.Process(input_buffer, output_buffer);
        REQUIRE(allocation_counter.Count() == 0);
    }
}

TEST_CASE("AudioProcessorChain reuses dirty intermediate storage for accumulating processors", "[processor_chain]")
{
    sfFDN::AudioProcessorChain chain(1U);

    auto first = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    first->SetGains(std::array{2.F, 2.F, 2.F, 2.F});
    auto second = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    second->SetGains(std::array{3.F, 3.F, 3.F, 3.F});
    auto merge = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge);
    merge->SetGains(std::array{1.F, 1.F, 1.F, 1.F});
    auto identity = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    identity->SetGains(std::array{1.F});

    REQUIRE(chain.AddProcessor(std::move(first)));
    REQUIRE(chain.AddProcessor(std::move(second)));
    REQUIRE(chain.AddProcessor(std::move(merge)));
    REQUIRE(chain.AddProcessor(std::move(identity)));

    std::array<float, 4> input = {1.F, 2.F, 3.F, 4.F};
    std::array<float, 1> output{};
    sfFDN::AudioBuffer const input_buffer(1U, 4U, input);
    sfFDN::AudioBuffer output_buffer(output);
    chain.Process(input_buffer, output_buffer);

    REQUIRE(output[0] == Catch::Approx(62.F));
}

TEST_CASE("AudioProcessorChain rejects channel mismatches without changing its contents", "[processor_chain]")
{
    sfFDN::AudioProcessorChain chain(8);
    auto split = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split);
    split->SetGains(std::array{1.f, 1.f});
    REQUIRE(chain.AddProcessor(std::move(split)));

    auto incompatible = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    incompatible->SetGains(std::array{1.f, 1.f, 1.f});
    REQUIRE_FALSE(chain.AddProcessor(std::move(incompatible)));
    REQUIRE(chain.GetProcessorCount() == 1);
    REQUIRE(chain.InputChannelCount() == 1);
    REQUIRE(chain.OutputChannelCount() == 2);
}

TEST_CASE("AudioProcessorChain writes single processor output directly", "[processor_chain]")
{
    constexpr uint32_t kBlockSize = 4;
    std::array<float, kBlockSize> input = {1.f, 2.f, 3.f, 4.f};
    std::array<float, kBlockSize> output = {-1.f, -1.f, -1.f, -1.f};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    sfFDN::AudioProcessorChain chain(kBlockSize);
    auto gain = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel);
    gain->SetGains(std::array{2.f});
    REQUIRE(chain.AddProcessor(std::move(gain)));

    chain.Process(input_buffer, output_buffer);
    REQUIRE(output == std::array{2.f, 4.f, 6.f, 8.f});
}