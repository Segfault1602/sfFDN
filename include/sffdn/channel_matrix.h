// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace sfFDN
{

/** @brief Applies a static dense matrix between planar audio channels.
 *
 * Coefficients are row-major with one row per output channel:
 * `coefficients[output_channel * input_channel_count + input_channel]`.
 * Processing overwrites every output sample. Input and output must occupy non-overlapping memory.
 * @ingroup AudioProcessors
 */
class ChannelMatrix final : public AudioProcessor
{
  public:
    explicit ChannelMatrix(const ChannelMatrixOptions& options);

    ChannelMatrix(const ChannelMatrix&) = delete;
    ChannelMatrix& operator=(const ChannelMatrix&) = delete;
    ChannelMatrix(ChannelMatrix&&) noexcept = default;
    ChannelMatrix& operator=(ChannelMatrix&&) noexcept = default;
    ~ChannelMatrix() override = default;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    void Clear() override;
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t input_channel_count_;
    uint32_t output_channel_count_;
    std::vector<float> coefficients_;
};

} // namespace sfFDN
