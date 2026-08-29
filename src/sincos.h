#pragma once

#include "sffdn/attributes.h"
#include "sine_table.h"

#include <bit>
#include <cmath>
#include <cstdint>

namespace sfFDN
{

static_assert(std::has_single_bit(kSineTableSize));
static_assert(kSineTableSize % 4U == 0U);
static_assert(kSineTable.size() == kSineTableSize + 1U);
static_assert(kSineTable[kSineTableSize] == kSineTable[0U]);

/** @brief Returns an interpolated sine-table value for a normalized phase.
 * @param phase The phase in cycles. Finite values are reduced to [0, 1).
 */
inline float SineTableLookup(float phase) noexcept SFFDN_NONBLOCKING
{
    phase -= std::floor(phase);
    const float index = phase * static_cast<float>(kSineTableSize);
    const auto table_index = static_cast<uint32_t>(index);
    const float fraction = index - static_cast<float>(table_index);
    const float a = kSineTable[table_index];
    const float b = kSineTable[table_index + 1U];
    return a + ((b - a) * fraction);
}

/** @brief Returns a unit sine/cosine pair from the lookup table.
 *
 * Component accuracy is guaranteed for finite angles in [-2π, 2π]. Callers with larger angles must range-reduce
 * them before calling this hot-path function. Finite inputs outside that range still produce a normalized pair, but
 * component accuracy is not guaranteed.
 */
inline void SinCosUnit(float radians, float& sin_out, float& cos_out) noexcept SFFDN_NONBLOCKING
{
    constexpr float kRadiansToCycles = 0.15915494309189533577f;
    constexpr uint32_t kTableMask = kSineTableSize - 1U;
    constexpr uint32_t kQuarterCycleOffset = kSineTableSize / 4U;

    const float cycles = radians * kRadiansToCycles;
    const float phase = cycles - std::floor(cycles);
    const float index = phase * static_cast<float>(kSineTableSize);
    const auto unmasked_index = static_cast<uint32_t>(index);
    const uint32_t sine_index = unmasked_index & kTableMask;
    const uint32_t cosine_index = (sine_index + kQuarterCycleOffset) & kTableMask;
    const float fraction = index - static_cast<float>(unmasked_index);

    const float sine_a = kSineTable[sine_index];
    const float sine_b = kSineTable[sine_index + 1U];
    sin_out = sine_a + ((sine_b - sine_a) * fraction);

    const float cosine_a = kSineTable[cosine_index];
    const float cosine_b = kSineTable[cosine_index + 1U];
    cos_out = cosine_a + ((cosine_b - cosine_a) * fraction);

    const float scale = 0.5f * (3.0f - ((sin_out * sin_out) + (cos_out * cos_out)));
    sin_out *= scale;
    cos_out *= scale;
}

} // namespace sfFDN
