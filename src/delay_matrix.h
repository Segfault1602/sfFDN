#pragma once

#include <mdspan>
#include <span>
#include <vector>

#include <delay.h>
#include <mixing_matrix.h>

namespace fdn
{
class DelayMatrix : public MixMat
{
  public:
    DelayMatrix(size_t N, std::span<const size_t> delays);

    void Clear();

    void SetDelays(std::span<size_t> delays);
    void SetMatrix(MixMat mixing_matrix);

    void Tick(std::span<const float> input, std::span<float> output) override;

    // Debug functions
    void DumpDelays() const;

  private:
    size_t N_;
    std::vector<Delay> delays_;
    MixMat mixing_matrix_;
};
} // namespace fdn