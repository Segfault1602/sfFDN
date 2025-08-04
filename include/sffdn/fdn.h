// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <span>

#include "audio_buffer.h"
#include "audio_processor.h"
#include "delaybank.h"
#include "feedback_matrix.h"
#include "filterbank.h"
#include "parallel_gains.h"
#include "schroeder_allpass.h"

namespace sfFDN
{

class FDN : public AudioProcessor
{
  public:
    FDN(uint32_t N, uint32_t block_size = 1, bool transpose = false);
    ~FDN() = default;

    void SetInputGains(std::unique_ptr<AudioProcessor> gains);
    void SetOutputGains(std::unique_ptr<AudioProcessor> gains);

    void SetInputGains(std::span<const float> gains);
    void SetOutputGains(std::span<const float> gains);

    void SetDirectGain(float gain);

    void SetFilterBank(std::unique_ptr<AudioProcessor> filter_bank);
    void SetDelays(const std::span<const uint32_t> delays);

    void SetFeedbackMatrix(std::unique_ptr<FeedbackMatrix> mixing_matrix);

    void SetTCFilter(std::unique_ptr<AudioProcessor> filter);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override
    {
        return 1;
    }

    uint32_t OutputChannelCount() const override
    {
        return 1;
    }

  private:
    void Tick(const AudioBuffer& input, AudioBuffer& output);
    void TickTranspose(const AudioBuffer& input, AudioBuffer& output);

    DelayBank delay_bank_;
    std::unique_ptr<AudioProcessor> filter_bank_;
    std::unique_ptr<FeedbackMatrix> mixing_matrix_;

    std::unique_ptr<AudioProcessor> input_gains_;
    std::unique_ptr<AudioProcessor> output_gains_;

    const uint32_t N_;
    const uint32_t block_size_;
    float direct_gain_;
    std::vector<float> feedback_;
    std::vector<float> temp_buffer_;

    std::unique_ptr<AudioProcessor> tc_filter_;

    bool transpose_;
};
} // namespace sfFDN