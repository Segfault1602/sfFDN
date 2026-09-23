#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

#include "audio_buffer_alias.h"
#include "passthrough.h"
#include "sffdn/audio_buffer.h"

TEST_CASE("PassThrough copies disjoint buffers", "[processor_chain]")
{
    constexpr std::array<float, 5> kExpected = {1.f, -2.f, 3.f, -4.f, 5.f};
    auto input = kExpected;
    std::array<float, 5> output{};
    sfFDN::AudioBuffer const input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    sfFDN::PassThrough pass_through;

    pass_through.Process(input_buffer, output_buffer);

    REQUIRE(output == kExpected);
}

TEST_CASE("PassThrough treats exact alias as a no-op", "[processor_chain]")
{
    constexpr std::array<float, 5> kExpected = {1.f, -2.f, 3.f, -4.f, 5.f};
    constexpr uint32_t kOffset = 3;
    constexpr uint32_t kStride = 12;
    std::array<float, kStride> samples{};
    std::ranges::copy(kExpected, samples.begin() + kOffset);
    sfFDN::AudioBuffer const parent_buffer(kStride, 1, samples);
    sfFDN::AudioBuffer alias_input = parent_buffer.Offset(kOffset, kExpected.size());
    sfFDN::AudioBuffer alias_output = parent_buffer.Offset(kOffset, kExpected.size());
    sfFDN::PassThrough pass_through;

    REQUIRE(sfFDN::ClassifyAudioBufferAlias(alias_input, alias_output) == sfFDN::AudioBufferAlias::Exact);

    // The no-op branch avoids overlapping-copy undefined behaviour without changing output values.
    pass_through.Process(alias_input, alias_output);
    std::array<float, kExpected.size()> output{};
    std::ranges::copy(alias_output.GetChannelSpan(0), output.begin());
    REQUIRE(output == kExpected);
}
