// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "delaybank.h"
#include "feedback_matrix.h"
#include "matrix_gallery.h"

namespace sfFDN
{

class FilterFeedbackMatrix : public FeedbackMatrix
{
  public:
    FilterFeedbackMatrix(uint32_t N);

    void Clear();
    // void SetDelays(std::span<uint32_t> delays);
    // void SetMatrices(std::span<ScalarFeedbackMatrix> mixing_matrices);

    void ConstructMatrix(std::span<uint32_t> delays, std::span<ScalarFeedbackMatrix> mixing_matrices);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override
    {
        return N_;
    }

    uint32_t OutputChannelCount() const override
    {
        return N_;
    }

    void PrintInfo() const;

  private:
    std::vector<DelayBank> delays_;
    std::vector<ScalarFeedbackMatrix> matrix_;
};

std::unique_ptr<FilterFeedbackMatrix> MakeFilterFeedbackMatrix(CascadedFeedbackMatrixInfo& info);

} // namespace sfFDN