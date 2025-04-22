#pragma once

#include <cstddef>
#include <span>

#include "delaybank.h"
#include "filterbank.h"
#include "mixing_matrix.h"
#include "schroeder_allpass.h"

namespace fdn
{
class FDN
{
  public:
    FDN(size_t N, size_t block_size = 1, bool transpose = false);
    ~FDN() = default;

    void Clear();

    void SetInputGains(const std::span<const float> gains);
    void SetOutputGains(const std::span<const float> gains);
    void SetDirectGain(float gain);

    FilterBank* GetFilterBank();
    DelayBank* GetDelayBank();

    void SetFeedbackMatrix(std::unique_ptr<FeedbackMatrix> mixing_matrix);
    FeedbackMatrix* GetMixingMatrix();

    void SetBypassAbsorption(bool bypass);

    void SetDelayModulation(float freq, float depth);

    void SetTCFilter(std::unique_ptr<Filter> filter);
    void SetSchroederSection(std::unique_ptr<SchroederAllpassSection> section);

    void Tick(const std::span<const float> input, std::span<float> output);

  private:
    void TickTranspose(const std::span<const float> input, std::span<float> output);

    DelayBank delay_bank_;
    FilterBank filter_bank_;
    std::unique_ptr<FeedbackMatrix> mixing_matrix_;

    const size_t N_;
    const size_t block_size_;
    std::vector<float> input_gain_;
    std::vector<float> output_gain_;
    float direct_gain_;
    std::vector<float> feedback_;
    std::vector<float> temp_buffer_;

    std::unique_ptr<Filter> tc_filter_;
    std::unique_ptr<SchroederAllpassSection> schroeder_section_;

    bool bypass_absorption_;
    bool transpose_;
};
} // namespace fdn