// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_processor.h"
#include "sffdn/types.h"

#include <memory>

namespace sfFDN
{
std::unique_ptr<AudioProcessor> CreateSingleChannelProcessor(const single_channel_processor_variant_t& config);
} // namespace sfFDN
