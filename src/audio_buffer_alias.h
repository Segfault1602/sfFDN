// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"
#include "sffdn/audio_buffer.h"

#include <cstdint>

namespace sfFDN
{
// Two buffers must either occupy non-overlapping memory extents or be the same logical view. The extent of a
// buffer runs from the first sample of channel 0 to the end of its last channel.
enum class AudioBufferAlias : uint8_t
{
    Disjoint,
    Exact,
    Invalid,
};

AudioBufferAlias ClassifyAudioBufferAlias(const AudioBuffer& first,
                                          const AudioBuffer& second) noexcept SFFDN_NONBLOCKING;
} // namespace sfFDN
