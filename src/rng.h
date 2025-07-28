#pragma once

#include <cstdint>

namespace sfFDN
{
/// @brief A simple random number generator using the xorshift algorithm. Not thread-safe.
uint32_t rng();
} // namespace sfFDN