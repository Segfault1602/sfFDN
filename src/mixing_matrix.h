#pragma once

#include <cstddef>
#include <span>

#include <Eigen/Core>

namespace fdn
{

class FeedbackMatrix
{
  public:
    virtual ~FeedbackMatrix() = default;
    virtual void Tick(const std::span<const float> input, std::span<float> output) = 0;
};

class MixMat : public FeedbackMatrix
{
  public:
    MixMat(size_t N = 4);
    virtual ~MixMat() = default;

    static MixMat Householder(size_t N);
    static MixMat Householder(std::span<const float> v);
    static MixMat Hadamard(size_t N);
    static MixMat Eye(size_t N);

    void SetMatrix(const std::span<const float> matrix);

    virtual void Tick(const std::span<const float> input, std::span<float> output) override;

    void Print() const;

    size_t GetSize() const
    {
        return N_;
    }

  private:
    size_t N_;
    Eigen::MatrixXf matrix_;
};
} // namespace fdn