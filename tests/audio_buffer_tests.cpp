#include <catch2/catch_test_macros.hpp>

#include <array>

#include "sffdn/audio_buffer.h"

TEST_CASE("AudioBuffer constructors and accessors alias backing storage")
{
    sfFDN::AudioBuffer const empty;
    REQUIRE(empty.SampleCount() == 0);
    REQUIRE(empty.ChannelCount() == 0);

    std::array<float, 12> storage{};
    for (auto i = 0u; i < storage.size(); ++i)
    {
        storage[i] = static_cast<float>(i + 1);
    }

    sfFDN::AudioBuffer mono(storage);
    REQUIRE(mono.SampleCount() == storage.size());
    REQUIRE(mono.ChannelCount() == 1);
    REQUIRE(mono.Data() == storage.data());
    mono.GetChannelSpan(0)[1] = -2.f;
    REQUIRE(storage[1] == -2.f);

    sfFDN::AudioBuffer buffer(4, 3, storage);
    const sfFDN::AudioBuffer& const_buffer = buffer;
    REQUIRE(const_buffer.Data() == storage.data());

    const auto const_span = const_buffer.GetChannelSpan(1);
    REQUIRE(const_span.data() == std::span(storage).subspan(4).data());
    REQUIRE(const_span.size() == 4);
    REQUIRE(const_span[2] == storage[6]);

    auto span = buffer.GetChannelSpan(2);
    span[1] = 42.f;
    REQUIRE(storage[9] == 42.f);

    auto channel_buffer = buffer.GetChannelBuffer(1);
    REQUIRE(channel_buffer.SampleCount() == 4);
    REQUIRE(channel_buffer.ChannelCount() == 1);
    REQUIRE(channel_buffer.Data() == std::span(storage).subspan(4).data());
    channel_buffer.GetChannelSpan(0)[0] = 24.f;
    REQUIRE(storage[4] == 24.f);

    const auto const_channel_buffer = const_buffer.GetChannelBuffer(2);
    REQUIRE(const_channel_buffer.Data() == std::span(storage).subspan(8).data());
    REQUIRE(const_channel_buffer.GetChannelSpan(0)[1] == 42.f);
}

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