#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "delay_matrix.h"
#include "mixing_matrix.h"

namespace fdn
{

class FilterFeedbackMatrix : public FeedbackMatrix
{
  public:
    FilterFeedbackMatrix(size_t N, size_t K);

    void Clear();
    void SetDelays(std::span<size_t> delays);
    void SetMatrices(std::span<MixMat> mixing_matrices);

    void Tick(std::span<const float> input, std::span<float> output) override;

    // Debug functions

    void DumpDelays() const;

  private:
    size_t N_;
    size_t K_;
    std::vector<DelayMatrix> stages_;

    MixMat last_mat_;
};
} // namespace fdn