// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "audio_processor.h"

namespace sfFDN
{
class FeedbackMatrix : public AudioProcessor
{
  public:
    FeedbackMatrix(uint32_t N)
        : N_(N) {};

    virtual ~FeedbackMatrix() = default;

    // virtual void Process(const AudioBuffer& input, AudioBuffer& output) = 0;

    uint32_t InputChannelCount() const override
    {
        return N_;
    }

    uint32_t OutputChannelCount() const override
    {
        return N_;
    }

  protected:
    uint32_t N_;
};

class ScalarFeedbackMatrix : public AudioProcessor
{
  public:
    ScalarFeedbackMatrix(uint32_t N = 4);
    virtual ~ScalarFeedbackMatrix() override;

    ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other);
    ScalarFeedbackMatrix& operator=(const ScalarFeedbackMatrix& other);

    static ScalarFeedbackMatrix Householder(uint32_t N);
    static ScalarFeedbackMatrix Householder(std::span<const float> v);
    static ScalarFeedbackMatrix Hadamard(uint32_t N);
    static ScalarFeedbackMatrix Eye(uint32_t N);

    bool SetMatrix(const std::span<const float> matrix);

    virtual void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void Print() const;

    uint32_t GetSize() const;

    float GetCoefficient(size_t row, size_t col) const;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

  private:
    class ScalarFeedbackMatrixImpl;
    std::unique_ptr<ScalarFeedbackMatrixImpl> impl_;
};

} // namespace sfFDN