#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "filter.h"

namespace fdn
{
class FilterBank
{
  public:
    FilterBank(size_t filterCount);
    ~FilterBank();

    void Clear();

    void SetFilter(size_t index, Filter* filter);
    void Tick(const std::span<const float> input, std::span<float> output);

  private:
    std::vector<Filter*> filters_;
};

} // namespace fdn