// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <memory>
#include <cstdint>
#include <span>

#include "audio_processor.h"

namespace sfFDN
{
class ScalarFeedbackMatrix : public AudioProcessor
{
  public:
    explicit ScalarFeedbackMatrix(uint32_t N = 4);
    ~ScalarFeedbackMatrix() override;

    ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other);
    ScalarFeedbackMatrix& operator=(const ScalarFeedbackMatrix& other);

    ScalarFeedbackMatrix(ScalarFeedbackMatrix&& other) noexcept;
    ScalarFeedbackMatrix& operator=(ScalarFeedbackMatrix&& other) noexcept;

    static ScalarFeedbackMatrix Householder(uint32_t N);
    static ScalarFeedbackMatrix Householder(std::span<const float> v);
    static ScalarFeedbackMatrix Hadamard(uint32_t N);
    static ScalarFeedbackMatrix Eye(uint32_t N);

    bool SetMatrix(const std::span<const float> matrix);

    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void Print() const;

    uint32_t GetSize() const;

    float GetCoefficient(uint32_t row, uint32_t col) const;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

  private:
    class ScalarFeedbackMatrixImpl;
    std::unique_ptr<ScalarFeedbackMatrixImpl> impl_;
};

} // namespace sfFDN