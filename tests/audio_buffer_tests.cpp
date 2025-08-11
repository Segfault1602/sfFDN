#include <catch2/catch_test_macros.hpp>

#include <array>

#include "sffdn/audio_buffer.h"
#include "sffdn/sffdn.h"

TEST_CASE("AudioBuffer_Offset")
{
    constexpr uint32_t frame_size = 128;
    constexpr uint32_t channels = 8;

    std::array<float, frame_size * channels> buffer{};

    for (uint32_t i = 0; i < channels; ++i)
    {
        for (uint32_t j = 0; j < frame_size; ++j)
        {
            buffer.at(i * frame_size + j) = static_cast<float>(j);
        }
    }

    sfFDN::AudioBuffer audio_buffer(frame_size, channels, buffer);
    REQUIRE(audio_buffer.SampleCount() == frame_size);
    REQUIRE(audio_buffer.ChannelCount() == channels);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < channels; ++i)
    {
        auto channel_span = audio_buffer.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j));
        }
    }

    constexpr uint32_t offset = 16;
    constexpr uint32_t new_frame_size = 32;

    auto offset_buffer = audio_buffer.Offset(offset, new_frame_size);
    REQUIRE(offset_buffer.SampleCount() == new_frame_size);
    REQUIRE(offset_buffer.ChannelCount() == channels);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < channels; ++i)
    {
        auto channel_span = offset_buffer.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j) + offset);
        }
    }

    auto twice_offset_buffer = offset_buffer.Offset(offset, new_frame_size);
    REQUIRE(twice_offset_buffer.SampleCount() == new_frame_size);
    REQUIRE(twice_offset_buffer.ChannelCount() == channels);

    auto twice_offset_from_original = audio_buffer.Offset(2 * offset, new_frame_size);
    REQUIRE(twice_offset_from_original.SampleCount() == new_frame_size);
    REQUIRE(twice_offset_from_original.ChannelCount() == channels);

    // Check that every channel contains the expected values
    for (uint32_t i = 0; i < channels; ++i)
    {
        auto channel_span = twice_offset_buffer.GetChannelSpan(i);
        auto channel_span2 = twice_offset_from_original.GetChannelSpan(i);
        for (uint32_t j = 0; j < channel_span.size(); ++j)
        {
            REQUIRE(channel_span[j] == static_cast<float>(j) + 2 * offset);
            REQUIRE(channel_span2[j] == static_cast<float>(j) + 2 * offset);
        }
    }
}