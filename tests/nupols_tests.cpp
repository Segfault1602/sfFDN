#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "signal_test_utils.h"
#include "test_utils.h"

TEST_CASE("PartitionedConvolver")
{
    constexpr uint32_t kBlockSize = 128;

    auto fir = sfFDNTest::CreateReferenceAbsorptionFir();

    sfFDN::PartitionedConvolver partitioned_convolver(kBlockSize, fir, 2);

    std::vector<float> input(fir.size(), 0.f);
    input[0] = 1.f;
    const auto output = sfFDNTest::RenderMonoBlocks(
        input, kBlockSize, 0, [&partitioned_convolver](std::span<float> input_block, std::span<float> output_block) {
            sfFDN::AudioBuffer const input_buffer(input_block);
            sfFDN::AudioBuffer output_buffer(output_block);
            partitioned_convolver.Process(input_buffer, output_buffer);
        });
    sfFDNTest::RequireSignalsClose(fir, output, 1e-5f, 90.0);
}

TEST_CASE("PartitionedConvolver_Noise")
{
    constexpr uint32_t kBlockSize = 128;

    auto fir = sfFDNTest::CreateReferenceAbsorptionFir();

    sfFDN::Fir fir_filter;
    fir_filter.SetCoefficients(fir);

    std::vector<float> input_chirp = ReadWavFile("./tests/data/chirp.wav");
    const uint32_t input_size = input_chirp.size();

    std::vector<float> filter_output(input_size, 0.f);
    sfFDN::AudioBuffer input_buffer(input_size, 1, input_chirp);
    sfFDN::AudioBuffer ref_output_buffer(input_size, 1, filter_output);

    fir_filter.Process(input_buffer, ref_output_buffer);

    sfFDN::PartitionedConvolver partitioned_convolver(kBlockSize, fir);
    const auto output = sfFDNTest::RenderMonoBlocks(
        input_chirp, kBlockSize, 0,
        [&partitioned_convolver](std::span<float> input_block, std::span<float> output_block) {
            sfFDN::AudioBuffer const input_buffer(input_block);
            sfFDN::AudioBuffer output_buffer(output_block);
            partitioned_convolver.Process(input_buffer, output_buffer);
        });
    sfFDNTest::RequireSignalsClose(filter_output, output, 1e-5f, 90.0);
}

TEST_CASE("PartitionedConvolver automatically selects a production partition schedule")
{
    constexpr uint32_t kBlockSize = 1024;
    std::vector<float> short_rir(48000, 0.f);
    std::vector<float> long_rir(96000, 0.f);
    short_rir[0] = 1.f;
    long_rir[0] = 1.f;

    sfFDN::PartitionedConvolver automatic_short(kBlockSize, short_rir);
    sfFDN::PartitionedConvolver explicit_short(kBlockSize, short_rir, 8);
    sfFDN::PartitionedConvolver automatic_long(kBlockSize, long_rir);
#if defined(__APPLE__) && defined(__aarch64__)
    sfFDN::PartitionedConvolver explicit_long(kBlockSize, long_rir, 16);
#else
    sfFDN::PartitionedConvolver explicit_long(kBlockSize, long_rir, 8);
#endif

    REQUIRE(automatic_short.GetShortInfo() == explicit_short.GetShortInfo());
    REQUIRE(automatic_long.GetShortInfo() == explicit_long.GetShortInfo());

    std::vector<float> input(kBlockSize, 0.f);
    std::vector<float> output(kBlockSize, 0.f);
    input[0] = 1.f;
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        automatic_long.Process(input_buffer, output_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);
}