#pragma once

#include <cstdint>

namespace sfFDN
{
/// @brief A simple random number generator using the xorshift algorithm
class RNG
{
  public:
    RNG() = default;
    RNG(uint32_t seed);

    constexpr uint32_t operator()() noexcept
    {
        state ^= (state << 13);
        state ^= (state >> 17);
        return (state ^= (state << 5));
    }

    constexpr float NextFloat() noexcept
    {
        auto next_uint = operator()();
        return (static_cast<float>(next_uint) * (2.f / static_cast<float>(UINT32_MAX))) - 1.f;
    }

  private:
    uint32_t state = 2463534242;
};

} // namespace sfFDN