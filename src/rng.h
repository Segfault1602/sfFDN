#pragma once

#include <cstdint>

namespace sfFDN
{
/// @brief A simple random number generator using the xorshift algorithm
class RNG
{
  public:
    RNG() = default;
    explicit RNG(uint32_t seed);

    constexpr float operator()() noexcept
    {
        const auto next_uint = NextUint();
        return (static_cast<float>(next_uint) * (2.f / static_cast<float>(UINT32_MAX))) - 1.f;
    }

    constexpr uint32_t NextUint() noexcept
    {
        state ^= (state << 13U);
        state ^= (state >> 17U);
        return (state ^= (state << 5U));
    }

  private:
    uint32_t state = 2463534242;
};

} // namespace sfFDN