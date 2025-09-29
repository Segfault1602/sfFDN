#include <catch2/catch_test_macros.hpp>

#include <array>

#include "sffdn/audio_buffer.h"

TEST_CASE("AudioBuffer_Offset")
{
    constexpr uint32_t kFrameSize = 128;
    constexpr uint32_t kChannelCount = 8;

    std::array<float, kFrameSize * kChannelCount> buffer{};

    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        for (uint32_t j = 0; j < kFrameSize; ++j)
        {
            buffer.at((i * kFrameSize) + j) = static_cast<float>(j);
        }
    }

    sfFDN::AudioBuffer audio_buffer(kFrameSize, kChannelCount, buffer);
    REQUIRE(audio_buffer.SampleCount() == kFrameSize);
    REQUIRE(audio_buffer.ChannelCount() == kChannelCount);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        auto channel_span = audio_buffer.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j));
        }
    }

    constexpr uint32_t kOffset = 16;
    constexpr uint32_t kNewFrameSize = 32;

    auto offset_buffer = audio_buffer.Offset(kOffset, kNewFrameSize);
    REQUIRE(offset_buffer.SampleCount() == kNewFrameSize);
    REQUIRE(offset_buffer.ChannelCount() == kChannelCount);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        auto channel_span = offset_buffer.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j) + kOffset);
        }
    }

    auto twice_offset_buffer = offset_buffer.Offset(kOffset, kNewFrameSize);
    REQUIRE(twice_offset_buffer.SampleCount() == kNewFrameSize);
    REQUIRE(twice_offset_buffer.ChannelCount() == kChannelCount);

    auto twice_offset_from_original = audio_buffer.Offset(2 * kOffset, kNewFrameSize);
    REQUIRE(twice_offset_from_original.SampleCount() == kNewFrameSize);
    REQUIRE(twice_offset_from_original.ChannelCount() == kChannelCount);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        auto channel_span = twice_offset_buffer.GetChannelSpan(i);
        auto channel_span2 = twice_offset_from_original.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j) + (2 * kOffset));
            REQUIRE(channel_span2[j] == static_cast<float>(j) + (2 * kOffset));
        }
    }
}