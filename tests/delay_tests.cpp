#include "doctest.h"

#include <array>
#include <iostream>
#include <numeric>
#include <sndfile.h>
#include <vector>

#include "delaya.h"
#include "delaybank.h"
#include <delay.h>

TEST_SUITE_BEGIN("DelayTests");

TEST_CASE("Delay")
{
    fdn::Delay delay(1, 10);

    std::vector<float> output;
    constexpr size_t iteration = 10;
    for (size_t i = 0; i < iteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr float expected_output[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (size_t i = 0; i < iteration; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.01));
    }
}

TEST_CASE("DelayA")
{
    fdn::DelayAllpass delay(1.5, 10);

    std::vector<float> output;
    constexpr size_t iteration = 10;
    for (size_t i = 0; i < iteration; ++i)
    {
        output.push_back(delay.Tick(i));
    }

    constexpr float expected_output[] = {0, 0, 0.33, 1.55, 2.48, 3.50, 4.49, 5.50, 6.50, 7.50};

    for (size_t i = 0; i < iteration; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.01));
    }
}

template <typename T>
void TestDelayBlock(float delay, size_t block_size, size_t max_delay)
{
    T delay_sample(delay, max_delay);

    std::vector<float> output_sample;
    for (size_t i = 0; i < block_size; ++i)
    {
        output_sample.push_back(delay_sample.Tick(i));
    }

    T delay_block(delay, max_delay);
    std::vector<float> input_block(block_size, 0.f);
    std::iota(input_block.begin(), input_block.end(), 0.f);

    std::vector<float> output_block(block_size, 0.f);

    fdn::AudioBuffer input_buffer(block_size, 1, input_block.data());
    fdn::AudioBuffer output_buffer(block_size, 1, output_block.data());

    delay_block.Process(input_buffer, output_buffer);

    for (size_t i = 0; i < block_size; ++i)
    {
        CHECK(output_block[i] == doctest::Approx(output_sample[i]));
    }
}

TEST_CASE("DelayBlock")
{
    TestDelayBlock<fdn::Delay>(1, 8, 10);
}

TEST_CASE("DelayABlock")
{
    TestDelayBlock<fdn::DelayAllpass>(1.5f, 8, 10);
}

TEST_CASE("DelayBank")
{
    constexpr size_t kNumDelay = 4;
    constexpr std::array<size_t, kNumDelay> delays = {2, 3, 4, 5};
    fdn::DelayBank delay_bank(delays, 10);

    std::vector<float> output;

    constexpr std::array<float, kNumDelay> impulse = {1, 1, 1, 1};
    std::array<float, 4> buffer = {0, 0, 0, 0};

    fdn::AudioBuffer impulse_buffer(1, kNumDelay, impulse.data());
    fdn::AudioBuffer buffer_audio(1, kNumDelay, buffer.data());

    delay_bank.Process(impulse_buffer, buffer_audio);
    for (auto& i : buffer)
    {
        output.push_back(i);
    }

    constexpr size_t iter = 9;
    for (size_t i = 0; i < iter; ++i)
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

    CHECK(output.size() == 40);
    for (size_t i = 0; i < output.size(); i += 4)
    {
        CHECK(output[i] == doctest::Approx(delay0_expected[i / 4]).epsilon(0.01));
        CHECK(output[i + 1] == doctest::Approx(delay1_expected[i / 4]).epsilon(0.01));
        CHECK(output[i + 2] == doctest::Approx(delay2_expected[i / 4]).epsilon(0.01));
        CHECK(output[i + 3] == doctest::Approx(delay3_expected[i / 4]).epsilon(0.01));
    }
}

TEST_CASE("DelayBankBlock")
{
    constexpr size_t kNumDelay = 4;
    constexpr size_t kBlockSize = 8;
    constexpr std::array<size_t, kNumDelay> delays = {2, 3, 4, 5};
    fdn::DelayBank delay_bank(delays, 10);

    std::vector<float> input(kNumDelay * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < kNumDelay; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kNumDelay * kBlockSize, 0.f);

    fdn::AudioBuffer input_buffer(kBlockSize, kNumDelay, input.data());
    fdn::AudioBuffer output_buffer(kBlockSize, kNumDelay, output.data());

    delay_bank.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> delay0_expected = {0, 0, 1, 0, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay1_expected = {0, 0, 0, 1, 0, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay2_expected = {0, 0, 0, 0, 1, 0, 0, 0};
    constexpr std::array<float, kBlockSize> delay3_expected = {0, 0, 0, 0, 0, 1, 0, 0};

    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[0 * kBlockSize + j] == doctest::Approx(delay0_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[1 * kBlockSize + j] == doctest::Approx(delay1_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[2 * kBlockSize + j] == doctest::Approx(delay2_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[3 * kBlockSize + j] == doctest::Approx(delay3_expected[j]).epsilon(0.01));
    }
}

TEST_SUITE_END(); // DelayTests
