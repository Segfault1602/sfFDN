#pragma once

#include <span>
#include <vector>

namespace fdn
{

class ParallelGains
{
  public:
    ParallelGains();

    void SetGains(std::span<const float> gains);

    void ProcessBlock(const std::span<const float> input, std::span<float> output);

  private:
    void ProcessBlockMultiplexed(const std::span<const float> input, std::span<float> output);
    void ProcessBlockDeMultiplexed(const std::span<const float> input, std::span<float> output);

    std::vector<float> gains_;
};
} // namespace fdn