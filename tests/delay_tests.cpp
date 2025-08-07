#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <limits>
#include <numeric>
#include <sndfile.h>
#include <vector>

#include "sffdn/sffdn.h"

TEST_CASE("Delay")
{
    sfFDN::Delay delay(1, 10);

    std::vector<float> output;
    constexpr uint32_t iteration = 10;
    for (uint32_t i = 0; i < iteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr float expected_output[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (uint32_t i = 0; i < iteration; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayTapOut")
{
    sfFDN::Delay delay(8, 10);

    std::vector<float> output;
    constexpr uint32_t iteration = 10;
    for (uint32_t i = 0; i < iteration; ++i)
    {
        delay.Tick(i);
        output.push_back(delay.TapOut(1));
    }

    constexpr float expected_output[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (uint32_t i = 0; i < iteration; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("ZeroDelay")
{
    sfFDN::Delay delay(0, 10);

    std::vector<float> output;
    constexpr uint32_t iteration = 10;
    for (uint32_t i = 0; i < iteration; ++i)
    {
        output.push_back(delay.Tick(i));

        REQUIRE(output[i] == delay.TapOut(0));
    }

    constexpr float expected_output[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (uint32_t i = 0; i < iteration; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayA")
{
    sfFDN::DelayAllpass delay(1.5, 10);

    std::vector<float> output;
    constexpr uint32_t iteration = 10;
    for (uint32_t i = 0; i < iteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr float expected_output[] = {0, 0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50};

    for (uint32_t i = 0; i < iteration; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-2));
    }
}

template <typename T>
void TestDelayBlock(float delay, uint32_t block_size, uint32_t max_delay)
{
    T delay_sample(delay, max_delay);

    std::vector<float> output_sample;
    for (uint32_t i = 0; i < block_size; ++i)
    {
        output_sample.push_back(delay_sample.Tick(i));
    }

    T delay_block(delay, max_delay);
    std::vector<float> input_block(block_size, 0.f);
    std::iota(input_block.begin(), input_block.end(), 0.f);

    std::vector<float> output_block(block_size, 0.f);

    sfFDN::AudioBuffer input_buffer(block_size, 1, input_block.data());
    sfFDN::AudioBuffer output_buffer(block_size, 1, output_block.data());

    delay_block.Process(input_buffer, output_buffer);

    for (uint32_t i = 0; i < block_size; ++i)
    {
        REQUIRE_THAT(output_block[i],
                     Catch::Matchers::WithinAbs(output_sample[i], std::numeric_limits<float>::epsilon()));
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
    constexpr std::array<uint32_t, kNumDelay> delays = {2, 3, 4, 5};
    sfFDN::DelayBank delay_bank(delays, 10);

    std::vector<float> output;

    std::array<float, kNumDelay> impulse = {1, 1, 1, 1};
    std::array<float, 4> buffer = {0, 0, 0, 0};

    sfFDN::AudioBuffer impulse_buffer(1, kNumDelay, impulse.data());
    sfFDN::AudioBuffer buffer_audio(1, kNumDelay, buffer.data());

    delay_bank.Process(impulse_buffer, buffer_audio);
    for (auto& i : buffer)
    {
        output.push_back(i);
    }

    constexpr uint32_t iter = 9;
    for (uint32_t i = 0; i < iter; ++i)
    {
        delay_bank.GetNextOutputs(buffer_audio);
        for (auto& i : buffer)
        {
            output.push_back(i);
        }

        buffer.fill(0);
        delay_bank.AddNextInputs(buffer_audio);
    }

    constexpr std::array<float, 10> delay0_expected = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> delay1_expected = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> delay2_expected = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, 10> delay3_expected = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    REQUIRE(output.size() == 40);
    for (uint32_t i = 0; i < output.size(); i += 4)
    {
        REQUIRE_THAT(output[i],
                     Catch::Matchers::WithinAbs(delay0_expected[i / 4], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 1],
                     Catch::Matchers::WithinAbs(delay1_expected[i / 4], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 2],
                     Catch::Matchers::WithinAbs(delay2_expected[i / 4], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(output[i + 3],
                     Catch::Matchers::WithinAbs(delay3_expected[i / 4], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankBlock")
{
    constexpr uint32_t kNumDelay = 4;
    constexpr uint32_t kBlockSize = 8;
    constexpr std::array<uint32_t, kNumDelay> delays = {2, 3, 4, 5};
    sfFDN::DelayBank delay_bank(delays, 10);

    std::vector<float> input(kNumDelay * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (uint32_t i = 0; i < kNumDelay; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kNumDelay * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kNumDelay, input.data());
    sfFDN::AudioBuffer output_buffer(kBlockSize, kNumDelay, output.data());

    delay_bank.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> delay0_expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay1_expected = {0, 0, 0, 1, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay2_expected = {0, 0, 0, 0, 1, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay3_expected = {0, 0, 0, 0, 0, 1, 0, 0};

    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[0 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(delay0_expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[1 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(delay1_expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[2 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(delay2_expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (uint32_t j = 0; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[3 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(delay3_expected[j], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("DelayBankProcess")
{
    constexpr uint32_t kBlockSize = 8;
    constexpr uint32_t kNumDelay = 4;
    constexpr std::array<uint32_t, kNumDelay> delays = {0, 1, 2, 3};
    sfFDN::DelayBank delay_bank(delays, kBlockSize);

    std::vector<float> output;

    std::array<float, kNumDelay * kBlockSize> impulse = {0.f};
    for (auto i = 0; i < kNumDelay; ++i)
    {
        impulse[i * kBlockSize] = 1.f;
    }
    std::array<float, kNumDelay * kBlockSize> buffer = {0.f};

    sfFDN::AudioBuffer impulse_buffer(kBlockSize, kNumDelay, impulse.data());
    sfFDN::AudioBuffer buffer_audio(kBlockSize, kNumDelay, buffer.data());

    delay_bank.Process(impulse_buffer, buffer_audio);

    constexpr std::array<float, kBlockSize> delay0_expected = {1, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay1_expected = {0, 1, 0, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay2_expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay3_expected = {0, 0, 0, 1, 0, 0, 0, 0};

    for (uint32_t i = 0; i < kBlockSize; ++i)
    {
        REQUIRE_THAT(buffer_audio.GetChannelSpan(0)[i],
                     Catch::Matchers::WithinAbs(delay0_expected[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(1)[i],
                     Catch::Matchers::WithinAbs(delay1_expected[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(2)[i],
                     Catch::Matchers::WithinAbs(delay2_expected[i], std::numeric_limits<float>::epsilon()));
        REQUIRE_THAT(buffer_audio.GetChannelSpan(3)[i],
                     Catch::Matchers::WithinAbs(delay3_expected[i], std::numeric_limits<float>::epsilon()));
    }
}