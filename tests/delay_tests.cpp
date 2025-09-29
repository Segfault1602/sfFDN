#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <sndfile.h>
#include <vector>

#include "sffdn/audio_buffer.h"
#include "sffdn/delay_utils.h"
#include "sffdn/sffdn.h"

namespace
{
template <typename T>
void TestDelayBlock(float delay, uint32_t block_size, uint32_t max_delay)
{
    T delay_sample(delay, max_delay);

    std::vector<float> output_sample;
    output_sample.reserve(block_size);
    for (uint32_t i = 0; i < block_size; ++i)
    {
        output_sample.push_back(delay_sample.Tick(i));
    }

    T delay_block(delay, max_delay);
    std::vector<float> input_block(block_size, 0.f);
    for (auto i = 0u; i < input_block.size(); ++i)
    {
        input_block[i] = i;
    }

    std::vector<float> output_block(block_size, 0.f);

    sfFDN::AudioBuffer input_buffer(block_size, 1, input_block);
    sfFDN::AudioBuffer output_buffer(block_size, 1, output_block);

    delay_block.Process(input_buffer, output_buffer);

    for (auto [out, expected] : std::views::zip(output_block, output_sample))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 1e-6));
    }
}
} // namespace

TEST_CASE("Delay")
{
    sfFDN::Delay delay(1, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    // for (uint32_t i = 0; i < iteration; ++i)
    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayTapOut")
{
    sfFDN::Delay delay(8, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        delay.Tick(i);
        output.push_back(delay.TapOut(1));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("ZeroDelay")
{
    sfFDN::Delay delay(0, 10);

    constexpr uint32_t kIteration = 10;
    std::vector<float> output;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));

        REQUIRE(output[i] == delay.TapOut(0));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayA")
{
    sfFDN::DelayAllpass delay(1.5, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }

    sfFDN::DelayAllpass delay_block(1.5, 32);

    std::vector<float> input_block(kIteration, 0.f);
    for (auto i = 0u; i < input_block.size(); ++i)
    {
        input_block[i] = i;
    }

    std::vector<float> output_block(kIteration, 0.f);

    sfFDN::AudioBuffer input_buffer(kIteration, 1, input_block);
    sfFDN::AudioBuffer output_buffer(kIteration, 1, output_block);

    delay_block.Process(input_buffer, output_buffer);

    for (auto [out, expected] : std::views::zip(output_block, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }
}

TEST_CASE("DelayA_MinDelay")
{
    sfFDN::DelayAllpass delay(0.5, 10);

    std::vector<float> output;
    constexpr uint32_t kIteration = 10;
    output.reserve(kIteration);
    for (uint32_t i = 0; i < kIteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr std::array<float, 10> kExpectedOutput = {0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50, 8.50};

    for (auto [out, expected] : std::views::zip(output, kExpectedOutput))
    {
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(expected, 0.01));
    }
}

TEST_CASE("DelayBlock")
{
    TestDelayBlock<sfFDN::Delay>(1, 8, 10);
}

TEST_CASE("DelayABlock")
{
    TestDelayBlock<sfFDN::DelayAllpass>(1.5f, 8, 10);
}

TEST_CASE("DelayBank")
{
    constexpr uint32_t kNumDelay = 4;
    constexpr std::array<uint32_t, kNumDelay> kDelays = {2, 3, 4, 5};
    sfFDN::DelayBank delay_bank(kDelays, 10);

    std::vector<float> output;

    std::array<float, kNumDelay> impulse = {1, 1, 1, 1};
    std::array<float, 4> buffer = {0, 0, 0, 0};

    sfFDN::AudioBuffer impulse_buffer(1, kNumDelay, impulse);
    sfFDN::AudioBuffer buffer_audio(1, kNumDelay, buffer);

    delay_bank.Process(impulse_buffer, buffer_audio);
    output.reserve(buffer.size());
    for (auto& i : buffer)
    {
        output.push_back(i);
    }

    constexpr uint32_t kIter = 9;
    for (uint32_t i = 0; i < kIter; ++i)
    {
        delay_bank.GetNextOutputs(buffer_audio);
        for (auto& i : buffer)
        {
            output.push_back(i);
        }

        buffer.fill(0);
        delay_bank.AddNextInputs(buffer_audio);
    }

    constexpr std::array<float, 10> kDelay0Expected = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay1Expected = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay2Expected = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> kDelay3Expected = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    REQUIRE(output.size() == 40);
    for (uint32_t i = 0; i < output.size(); i += 4)
    {
        REQUIRE_THAT(output[i],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 1],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 2],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 3],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(i / 4), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankBlock")
{
    constexpr uint32_t kNumDelay = 4;
    constexpr uint32_t kBlockSize = 8;
    constexpr std::array<uint32_t, kNumDelay> kDelays = {2, 3, 4, 5};
    sfFDN::DelayBank delay_bank(kDelays, 10);

    std::vector<float> input(kNumDelay * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (uint32_t i = 0; i < kNumDelay; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kNumDelay * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kNumDelay, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kNumDelay, output);

    delay_bank.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> kDelay0Expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay1Expected = {0, 0, 0, 1, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay2Expected = {0, 0, 0, 0, 1, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay3Expected = {0, 0, 0, 0, 0, 1, 0, 0};

    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output_buffer.GetChannelSpan(0)[j],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(1)[j],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(2)[j],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(j), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output_buffer.GetChannelSpan(3)[j],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(j), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankProcess")
{
    constexpr uint32_t kBlockSize = 8;
    constexpr uint32_t kNumDelay = 4;
    constexpr std::array<uint32_t, kNumDelay> kDelays = {0, 1, 2, 3};
    sfFDN::DelayBank delay_bank(kDelays, kBlockSize);

    std::vector<float> output;

    std::array<float, kNumDelay * kBlockSize> impulse = {0.f};
    for (auto i = 0u; i < kNumDelay; ++i)
    {
        impulse.at(i * kBlockSize) = 1.f;
    }
    std::array<float, kNumDelay * kBlockSize> buffer = {0.f};

    sfFDN::AudioBuffer impulse_buffer(kBlockSize, kNumDelay, impulse);
    sfFDN::AudioBuffer buffer_audio(kBlockSize, kNumDelay, buffer);

    delay_bank.Process(impulse_buffer, buffer_audio);

    constexpr std::array<float, kBlockSize> kDelay0Expected = {1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay1Expected = {0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay2Expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kDelay3Expected = {0, 0, 0, 1, 0, 0, 0, 0};

    for (uint32_t i = 0; i < kBlockSize; ++i)
    {
        REQUIRE_THAT(buffer_audio.GetChannelSpan(0)[i],
                     Catch::Matchers::WithinAbs(kDelay0Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(1)[i],
                     Catch::Matchers::WithinAbs(kDelay1Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(2)[i],
                     Catch::Matchers::WithinAbs(kDelay2Expected.at(i), std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(3)[i],
                     Catch::Matchers::WithinAbs(kDelay3Expected.at(i), std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayLengths")
{
    SKIP();
    auto delays = sfFDN::GetDelayLengths(16, 4500, 12000, sfFDN::DelayLengthType::SteamAudio, 0);

    for (auto d : delays)
    {
        std::cout << d << " ";
    }
    std::cout << "\n";

    delays = sfFDN::GetDelayLengthsFromMean(8, 20.f, 2.8f, 48000);
    for (auto d : delays)
    {
        std::cout << d << " ";
    }
    std::cout << "\n";
}