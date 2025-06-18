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
    FeedbackMatrix(size_t N)
        : N_(N) {};

    virtual ~FeedbackMatrix() = default;

    // virtual void Process(const AudioBuffer& input, AudioBuffer& output) = 0;

    size_t InputChannelCount() const override
    {
        return N_;
    }

    size_t OutputChannelCount() const override
    {
        return N_;
    }

  protected:
    size_t N_;
};

class ScalarFeedbackMatrix : public FeedbackMatrix
{
  public:
    ScalarFeedbackMatrix(size_t N = 4);
    virtual ~ScalarFeedbackMatrix() = default;

    static ScalarFeedbackMatrix Householder(size_t N);
    static ScalarFeedbackMatrix Householder(std::span<const float> v);
    static ScalarFeedbackMatrix Hadamard(size_t N);
    static ScalarFeedbackMatrix Eye(size_t N);

    void SetMatrix(const std::span<const float> matrix);

    virtual void Process(const AudioBuffer& input, AudioBuffer& output) override;

    void Print() const;

    size_t GetSize() const
    {
        return N_;
    }

  private:
    Eigen::MatrixXf matrix_;
};

} // namespace sfFDN