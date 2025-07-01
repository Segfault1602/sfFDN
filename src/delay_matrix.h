#pragma once

#include <mdspan>
#include <span>
#include <vector>

#include "delaybank.h"
#include <delay.h>
#include <feedback_matrix.h>

namespace sfFDN
{
class DelayMatrix : public FeedbackMatrix
{
  public:
    DelayMatrix(uint32_t N, std::span<const uint32_t> delays);

    void Clear();

    void SetDelays(std::span<uint32_t> delays);
    void SetMatrix(ScalarFeedbackMatrix mixing_matrix);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

  private:
    DelayBank delays_;
    ScalarFeedbackMatrix mixing_matrix_;
};
} // namespace sfFDN