// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "audio_processor.h"
#include "matrix_gallery.h"

namespace sfFDN
{
class ScalarFeedbackMatrix : public AudioProcessor
{
  public:
    ScalarFeedbackMatrix(uint32_t order = 4, ScalarMatrixType type = ScalarMatrixType::Identity);
    ScalarFeedbackMatrix(uint32_t order, std::span<const float> matrix);

    ~ScalarFeedbackMatrix() override;

    ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other);
    ScalarFeedbackMatrix& operator=(const ScalarFeedbackMatrix& other);

    ScalarFeedbackMatrix(ScalarFeedbackMatrix&& other) noexcept;
    ScalarFeedbackMatrix& operator=(ScalarFeedbackMatrix&& other) noexcept;

    bool SetMatrix(const std::span<const float> matrix);
    bool GetMatrix(std::span<float> matrix) const;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    void Print() const;

    uint32_t GetSize() const;

    float GetCoefficient(uint32_t row, uint32_t col) const;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    void Clear() override;

    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class ScalarFeedbackMatrixImpl;
    std::unique_ptr<ScalarFeedbackMatrixImpl> impl_;
};

} // namespace sfFDN