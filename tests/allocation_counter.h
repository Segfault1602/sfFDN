#pragma once

#include <cstddef>

namespace sfFDNTest
{

class ScopedAllocationCounter
{
  public:
    ScopedAllocationCounter() noexcept;
    ~ScopedAllocationCounter();

    ScopedAllocationCounter(const ScopedAllocationCounter&) = delete;
    ScopedAllocationCounter& operator=(const ScopedAllocationCounter&) = delete;

    [[nodiscard]] size_t Count() const noexcept;
};

} // namespace sfFDNTest
