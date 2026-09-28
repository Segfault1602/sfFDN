// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_processor.h"
#include "sffdn/types.h"

#include <memory>

namespace sfFDN
{
std::unique_ptr<AudioProcessor> CreateSingleChannelProcessor(const single_channel_processor_variant_t& config);

// Builds a multichannel bank. When every channel is a CascadedBiquadsOptions with the same nonzero stage count, the
// channels are evaluated together by a SIMD IIRFilterBank; otherwise each channel gets its own processor in a
// FilterBank.
std::unique_ptr<AudioProcessor> CreateMultichannelProcessor(const MultichannelProcessorOptions& options);
} // namespace sfFDN
