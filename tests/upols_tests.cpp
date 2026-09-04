#include <catch2/catch_test_macros.hpp>

#include <span>
#include <vector>

#include "upols.h"

#include "signal_test_utils.h"
#include "test_utils.h"

TEST_CASE("UPOLS reproduces the reference impulse response", "[convolution]")
{
    constexpr uint32_t kBlockSize = 32;

    auto fir = sfFDNTest::CreateReferenceAbsorptionFir();

    sfFDN::UPOLS upols;
    REQUIRE(upols.Initialize(kBlockSize, fir));

    std::vector<float> input(fir.size(), 0.f);
    input[0] = 1.f;
    const auto output = sfFDNTest::RenderMonoBlocks(
        input, kBlockSize, 0, [&upols](std::span<float> input_block, std::span<float> output_block) {
            upols.Process(input_block, output_block);
        });
    sfFDNTest::RequireSignalsClose(fir, output, 1e-6f, 100.0);
}

TEST_CASE("UPOLS matches Fir on a chirp", "[convolution]")
{
    constexpr uint32_t kBlockSize = 128;

    auto fir = sfFDNTest::CreateReferenceAbsorptionFir();

    std::vector<float> input_chirp = ReadWavFile("./tests/data/chirp.wav");
    const uint32_t input_size = input_chirp.size();

    std::vector<float> filter_output(input_size, 0.f);
    // Filter the input noise with the IIR filter
    sfFDN::AudioBuffer input_buffer(input_size, 1, input_chirp);
    sfFDN::AudioBuffer ref_output_buffer(input_size, 1, filter_output);

    sfFDN::Fir fir_filter;
    fir_filter.SetCoefficients(fir);
    fir_filter.Process(input_buffer, ref_output_buffer);

    sfFDN::UPOLS upols;
    REQUIRE(upols.Initialize(kBlockSize, fir));

    const auto output = sfFDNTest::RenderMonoBlocks(
        input_chirp, kBlockSize, 0, [&upols](std::span<float> input_block, std::span<float> output_block) {
            upols.Process(input_block, output_block);
        });
    sfFDNTest::RequireSignalsClose(filter_output, output, 1e-5f, 90.0);
}