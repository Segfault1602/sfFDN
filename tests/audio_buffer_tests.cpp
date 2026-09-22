#include <catch2/catch_test_macros.hpp>

#include <array>

#include "audio_buffer_alias.h"
#include "sffdn/audio_buffer.h"

TEST_CASE("AudioBuffer aliases backing storage through constructors and accessors", "[audio_buffer]")
{
    sfFDN::AudioBuffer const empty;
    REQUIRE(empty.SampleCount() == 0);
    REQUIRE(empty.ChannelCount() == 0);
    REQUIRE(empty.IsPacked());

    std::array<float, 12> storage{};
    for (auto i = 0u; i < storage.size(); ++i)
    {
        storage[i] = static_cast<float>(i + 1);
    }

    sfFDN::AudioBuffer mono(storage);
    REQUIRE(mono.SampleCount() == storage.size());
    REQUIRE(mono.ChannelCount() == 1);
    REQUIRE(mono.ChannelStride() == storage.size());
    REQUIRE(mono.IsPacked());
    REQUIRE(mono.Data() == storage.data());
    mono.GetChannelSpan(0)[1] = -2.f;
    REQUIRE(storage[1] == -2.f);

    sfFDN::AudioBuffer buffer(4, 3, storage);
    const sfFDN::AudioBuffer& const_buffer = buffer;
    REQUIRE(buffer.ChannelStride() == 4);
    REQUIRE(buffer.IsPacked());
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

    const auto shortened = buffer.Offset(0, 2);
    REQUIRE(shortened.ChannelStride() == 4);
    REQUIRE_FALSE(shortened.IsPacked());

    const auto shortened_mono = mono.Offset(0, 2);
    REQUIRE(shortened_mono.ChannelStride() == storage.size());
    REQUIRE(shortened_mono.SampleCount() == 2);
    REQUIRE(shortened_mono.IsPacked());

    const auto offset_shortened_mono = mono.Offset(1, 2);
    REQUIRE(offset_shortened_mono.Data() == storage.data());
    REQUIRE_FALSE(offset_shortened_mono.IsPacked());
}

TEST_CASE("AudioBuffer Offset returns offset channel data", "[audio_buffer]")
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

TEST_CASE("AudioBuffer alias classifier distinguishes logical relationships", "[audio_buffer]")
{
    std::array<float, 48> storage{};
    std::array<float, 24> separate_storage{};
    const sfFDN::AudioBuffer parent(8, 3, storage);
    const sfFDN::AudioBuffer copy = parent;
    const sfFDN::AudioBuffer independent(8, 3, storage);
    const sfFDN::AudioBuffer separate(8, 3, separate_storage);
    const sfFDN::AudioBuffer fewer_channels(8, 2, std::span(storage).first(16));
    const sfFDN::AudioBuffer separate_fewer_channels(8, 2, separate_storage);
    const sfFDN::AudioBuffer different_stride = sfFDN::AudioBuffer(16, 3, storage).Offset(0, 8);
    const sfFDN::AudioBuffer mono_different_stride =
        sfFDN::AudioBuffer(16, 1, std::span(storage).subspan(8)).Offset(0, 8);
    const sfFDN::AudioBuffer default_buffer;
    const sfFDN::AudioBuffer zero_channels(8, 0, storage);
    const sfFDN::AudioBuffer zero_frame(0, 3, storage);

    const auto require_relationship = [](const sfFDN::AudioBuffer& first, const sfFDN::AudioBuffer& second,
                                         sfFDN::AudioBufferAlias expected) {
        REQUIRE(sfFDN::ClassifyAudioBufferAlias(first, second) == expected);
        REQUIRE(sfFDN::ClassifyAudioBufferAlias(second, first) == expected);
    };

    require_relationship(parent, copy, sfFDN::AudioBufferAlias::Exact);
    require_relationship(parent, independent, sfFDN::AudioBufferAlias::Exact);
    require_relationship(parent, separate, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(parent, fewer_channels, sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent, separate_fewer_channels, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(parent, parent.GetChannelBuffer(1), sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent, different_stride, sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent.GetChannelBuffer(1), mono_different_stride, sfFDN::AudioBufferAlias::Exact);
    require_relationship(parent.Offset(0, 2), parent.Offset(2, 2), sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(parent.Offset(0, 2), parent.Offset(4, 2), sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(parent, parent.Offset(0, 4), sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent.Offset(1, 3), parent.Offset(2, 3), sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent.Offset(1, 6), parent.Offset(2, 2), sfFDN::AudioBufferAlias::Partial);
    require_relationship(parent.Offset(0, 2), parent.Offset(0, 3), sfFDN::AudioBufferAlias::Partial);

    require_relationship(default_buffer, parent, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(zero_channels, parent, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(zero_channels, zero_channels, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(zero_frame, parent, sfFDN::AudioBufferAlias::Disjoint);
    require_relationship(zero_frame, zero_frame, sfFDN::AudioBufferAlias::Disjoint);
}