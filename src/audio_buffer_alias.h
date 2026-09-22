// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"
#include "sffdn/audio_buffer.h"

#include <cstdint>

namespace sfFDN
{
enum class AudioBufferAlias : uint8_t
{
    Disjoint,
    Exact,
    Partial,
};

AudioBufferAlias ClassifyAudioBufferAlias(const AudioBuffer& first,
                                          const AudioBuffer& second) noexcept SFFDN_NONBLOCKING;
} // namespace sfFDN
