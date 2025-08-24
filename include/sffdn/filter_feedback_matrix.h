// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "delaybank.h"
#include "feedback_matrix.h"
#include "matrix_gallery.h"

namespace sfFDN
{

class FilterFeedbackMatrix : public AudioProcessor
{
  public:
    FilterFeedbackMatrix(uint32_t N);

    void ConstructMatrix(std::span<const uint32_t> delays, std::span<const ScalarFeedbackMatrix> mixing_matrices);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    uint32_t InputChannelCount() const override
    {
        return N_;
    }

    uint32_t OutputChannelCount() const override
    {
        return N_;
    }

    void Clear() override;

    void PrintInfo() const;

    // TODO: this is just for the GUI in FDNSandbox
    bool GetFirstMatrix(std::span<float> matrix) const;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t N_;
    std::vector<DelayBank> delays_;
    std::vector<ScalarFeedbackMatrix> matrix_;
};

std::unique_ptr<FilterFeedbackMatrix> MakeFilterFeedbackMatrix(const CascadedFeedbackMatrixInfo& info);

} // namespace sfFDN