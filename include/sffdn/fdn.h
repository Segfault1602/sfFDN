// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delaybank.h"

namespace sfFDN
{

class FDN : public AudioProcessor
{
  public:
    FDN(uint32_t N, uint32_t block_size = 0, bool transpose = false);
    ~FDN() = default;

    FDN(const FDN&) = delete;
    FDN& operator=(const FDN&) = delete;

    FDN(FDN&&) noexcept;
    FDN& operator=(FDN&&) noexcept;

    bool SetInputGains(std::unique_ptr<AudioProcessor> gains);
    bool SetOutputGains(std::unique_ptr<AudioProcessor> gains);

    bool SetInputGains(std::span<const float> gains);
    bool SetOutputGains(std::span<const float> gains);

    void SetDirectGain(float gain);

    bool SetFilterBank(std::unique_ptr<AudioProcessor> filter_bank);
    bool SetDelays(const std::span<const uint32_t> delays);

    bool SetFeedbackMatrix(std::unique_ptr<AudioProcessor> mixing_matrix);

    bool SetTCFilter(std::unique_ptr<AudioProcessor> filter);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override
    {
        return 1;
    }

    uint32_t OutputChannelCount() const override
    {
        return 1;
    }

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    void TickInternal(const AudioBuffer& input, AudioBuffer& output);
    void Tick(const AudioBuffer& input, AudioBuffer& output);
    void TickTranspose(const AudioBuffer& input, AudioBuffer& output);
    void TickTransposeInternal(const AudioBuffer& input, AudioBuffer& output);

    DelayBank delay_bank_;
    std::unique_ptr<AudioProcessor> filter_bank_;
    std::unique_ptr<AudioProcessor> mixing_matrix_;

    std::unique_ptr<AudioProcessor> input_gains_;
    std::unique_ptr<AudioProcessor> output_gains_;

    uint32_t N_;
    uint32_t block_size_;
    float direct_gain_;

    std::vector<float> feedback_;
    std::vector<float> temp_buffer_;

    std::unique_ptr<AudioProcessor> tc_filter_;

    bool transpose_;
};
} // namespace sfFDN