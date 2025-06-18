#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "delay_matrix.h"
#include "feedback_matrix.h"

namespace sfFDN
{

class FilterFeedbackMatrix : public FeedbackMatrix
{
  public:
    FilterFeedbackMatrix(size_t N);

    void Clear();
    // void SetDelays(std::span<size_t> delays);
    // void SetMatrices(std::span<ScalarFeedbackMatrix> mixing_matrices);

    void ConstructMatrix(std::span<size_t> delays, std::span<ScalarFeedbackMatrix> mixing_matrices);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    size_t InputChannelCount() const override
    {
        return N_;
    }

    size_t OutputChannelCount() const override
    {
        return N_;
    }

  private:
    std::vector<DelayMatrix> stages_;
    ScalarFeedbackMatrix last_mat_;
};
} // namespace sfFDN