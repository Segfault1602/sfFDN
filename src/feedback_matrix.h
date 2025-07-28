#pragma once

#include <cstddef>
#include <span>

#include <Eigen/Core>

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

class ScalarFeedbackMatrix : public FeedbackMatrix
{
  public:
    ScalarFeedbackMatrix(uint32_t N = 4);
    virtual ~ScalarFeedbackMatrix() = default;

    static ScalarFeedbackMatrix Householder(uint32_t N);
    static ScalarFeedbackMatrix Householder(std::span<const float> v);
    static ScalarFeedbackMatrix Hadamard(uint32_t N);
    static ScalarFeedbackMatrix Eye(uint32_t N);

    void SetMatrix(const std::span<const float> matrix);

    virtual void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void Print() const;

    uint32_t GetSize() const
    {
        return N_;
    }

  private:
    Eigen::MatrixXf matrix_;
    std::vector<float> matrix_coeffs_;
};

} // namespace sfFDN