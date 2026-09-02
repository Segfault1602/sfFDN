// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>

namespace sfFDN
{

/** @brief A single-channel processor that copies its input to its output.
 *
 * This exists to fill the slots of a FilterBank whose channels are only partially processed, so that the bank keeps
 * one processor per channel and the channel indices of the bank still line up with the channels of the FDN.
 *
 * Internal: not part of the public API.
 */
class PassThrough : public AudioProcessor
{
  public:
    PassThrough() = default;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == 1);
        assert(output.ChannelCount() == 1);

        const auto in_span = input.GetChannelSpan(0);
        const auto out_span = output.GetChannelSpan(0);
        std::ranges::copy(in_span, out_span.begin());
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return 1;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return 1;
    }

    void Clear() override
    {
    }

    std::unique_ptr<AudioProcessor> Clone() const override
    {
        return std::make_unique<PassThrough>(*this);
    }
};

} // namespace sfFDN
