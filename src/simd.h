// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include <xsimd/xsimd.hpp>

namespace sfFDN::simd
{

namespace detail
{
static_assert(xsimd::default_arch::supported(),
              "sfFDN requires a SIMD instruction set supported by xsimd; check the target's compiler flags");

using Arch = xsimd::default_arch;
} // namespace detail

/**
 * @brief A single-precision vector for the target's widest supported SIMD instruction set.
 *
 * The abstraction is intentionally minimal: only the operations required by the DSP kernels in this library are
 * provided. Every operation is branch-free and allocation-free so that callers remain real-time safe. Callers must not
 * assume a particular width; use kWidth.
 */
using Vec = xsimd::batch<float, detail::Arch>;
using IntVec = xsimd::batch<int32_t, detail::Arch>;

inline constexpr size_t kWidth = Vec::size;
// xsimd does not expose the size of the vector register file. The dense matrix kernel sizes its register tiles from
// this value, so it only needs to be right about whether 32 registers are available. xsimd feature macros are always
// defined as 0 or 1 and must be tested with #if.
#if XSIMD_WITH_SSE2 && !(defined(__x86_64__) || defined(_M_X64))
inline constexpr size_t kRegisterCount = 8;
#elif XSIMD_WITH_AVX512F || XSIMD_WITH_NEON64 || XSIMD_WITH_SVE || XSIMD_WITH_RVV
inline constexpr size_t kRegisterCount = 32;
#else
inline constexpr size_t kRegisterCount = 16;
#endif

static_assert(IntVec::size == kWidth);

inline Vec Load(const float* p) noexcept SFFDN_NONBLOCKING
{
    return Vec::load_unaligned(p);
}

inline void Store(float* p, Vec v) noexcept SFFDN_NONBLOCKING
{
    v.store_unaligned(p);
}

inline Vec Splat(float x) noexcept SFFDN_NONBLOCKING
{
    return xsimd::broadcast<float, detail::Arch>(x);
}

inline Vec Zero() noexcept SFFDN_NONBLOCKING
{
    return Splat(0.f);
}

inline Vec Add(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return a + b;
}

inline Vec Sub(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return a - b;
}

inline Vec Mul(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return a * b;
}

/** @brief Returns a * b + c, fused where the target supports it. */
inline Vec MulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return xsimd::fma(a, b, c);
}

/** @brief Returns c - a * b, fused where the target supports it. */
inline Vec NegMulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
#if XSIMD_WITH_NEON && !XSIMD_WITH_SVE && defined(__ARM_FEATURE_FMA)
    // xsimd has no NEON fnma, and its generic -a * b + c is split across operator calls, so the compiler cannot fuse
    // it; fma(-a, b, c) still costs a separate fneg. vfmsq_f32 is a single fmls/vfms, matching the pre-xsimd code.
    // NOLINTNEXTLINE(portability-simd-intrinsics)
    return Vec(vfmsq_f32(c, a, b));
#else
    return xsimd::fnma(a, b, c);
#endif
}

/** @brief Rounds toward negative infinity. Requires |x| < 2^31 in every lane. */
inline Vec Floor(Vec x) noexcept SFFDN_NONBLOCKING
{
    // xsimd 14.3.0 has a native floor only on SSE4.1, AVX, AVX-512, WASM and VSX. Its generic fallback handles
    // arbitrary magnitudes with an extra abs, compare and select, which cost the SSE2 oscillator ~10%; the
    // precondition allows a plain truncate-and-correct instead, and AArch64 has a single-instruction floor.
    // Selection is done by the preprocessor: function effect analysis also inspects discarded if-constexpr branches in
    // non-template functions, and cannot see into the xsimd templates they would have instantiated.
#if XSIMD_WITH_SSE4_1 || XSIMD_WITH_AVX || XSIMD_WITH_AVX512F || XSIMD_WITH_WASM || XSIMD_WITH_VSX
    return xsimd::floor(x);
#elif XSIMD_WITH_NEON64 && !XSIMD_WITH_SVE
    // NOLINTNEXTLINE(portability-simd-intrinsics)
    return Vec(vrndmq_f32(x));
#else
    const Vec truncated = xsimd::batch_cast<float>(xsimd::batch_cast<int32_t>(x));
    return xsimd::select(truncated > x, truncated - Vec(1.f), truncated);
#endif
}

/** @brief Converts to integers, truncating toward zero. Requires every lane to be representable as int32_t. */
inline IntVec ToInt(Vec x) noexcept SFFDN_NONBLOCKING
{
    return xsimd::batch_cast<int32_t>(x);
}

inline Vec ToFloat(IntVec x) noexcept SFFDN_NONBLOCKING
{
    return xsimd::batch_cast<float>(x);
}

inline IntVec Min(IntVec x, int32_t maximum) noexcept SFFDN_NONBLOCKING
{
    return xsimd::min(x, IntVec(maximum));
}

// xsimd has a hardware gather only on AVX2/AVX-512, SVE and RVV. Elsewhere its generic gather extracts and inserts one
// lane at a time, a serial dependency chain that made the SSE2 oscillator ~30% slower. Extracting the indices once
// and constructing the vector from every lane lets the compiler combine the lanes as a tree instead.
#if XSIMD_WITH_AVX2 || XSIMD_WITH_SVE || XSIMD_WITH_RVV
#define SFFDN_SIMD_NATIVE_GATHER 1
#else
#define SFFDN_SIMD_NATIVE_GATHER 0
#endif

namespace detail
{
using IndexLanes = std::array<int32_t, IntVec::size>;

inline IndexLanes ToIndexLanes(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
    IndexLanes lanes{};
    indices.store_unaligned(lanes.data());
#ifndef NDEBUG
    for (const int32_t index : lanes)
    {
        assert(index >= 0 && static_cast<size_t>(index) < values.size());
    }
#else
    static_cast<void>(values);
#endif
    return lanes;
}

template <size_t... Lane>
Vec GatherLanes(std::span<const float> values, const IndexLanes& lanes,
                std::index_sequence<Lane...> /*unused*/) noexcept SFFDN_NONBLOCKING
{
    return Vec(values[static_cast<size_t>(lanes[Lane])]...);
}
} // namespace detail

/** @brief Returns values[indices[i]] in lane i. Every index must be within @p values. */
inline Vec Gather(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
#if SFFDN_SIMD_NATIVE_GATHER
#ifndef NDEBUG
    static_cast<void>(detail::ToIndexLanes(values, indices));
#endif
    return Vec::gather(values.data(), indices);
#else
    return detail::GatherLanes(values, detail::ToIndexLanes(values, indices), std::make_index_sequence<kWidth>{});
#endif
}

struct AdjacentGather
{
    Vec lower;
    Vec upper;
};

/** @brief Returns values[indices[i]] in lower and values[indices[i] + 1] in upper. */
inline AdjacentGather GatherAdjacent(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
    // Temporary A/B switch: the pre-xsimd NEON implementation, which replaces two four-lane gathers with four pair
    // loads and an unzip. vuzp1q/vuzp2q_f32 are AArch64-only, and SVE builds use a different register type.
#if defined(SFFDN_SIMD_NEON_PAIR_GATHER) && XSIMD_WITH_NEON64 && !XSIMD_WITH_SVE
    // NOLINTBEGIN(portability-simd-intrinsics)
    static_assert(kWidth == 4);
    const int32x4_t lanes = indices;
    const float32x2_t pair0 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(lanes, 0))]);
    const float32x2_t pair1 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(lanes, 1))]);
    const float32x2_t pair2 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(lanes, 2))]);
    const float32x2_t pair3 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(lanes, 3))]);
    const float32x4_t low = vcombine_f32(pair0, pair1);
    const float32x4_t high = vcombine_f32(pair2, pair3);
    return {.lower = Vec(vuzp1q_f32(low, high)), .upper = Vec(vuzp2q_f32(low, high))};
    // NOLINTEND(portability-simd-intrinsics)
#elif SFFDN_SIMD_NATIVE_GATHER
    return {.lower = Gather(values, indices), .upper = Gather(values.subspan(1), indices)};
#else
    const detail::IndexLanes lanes = detail::ToIndexLanes(values.first(values.size() - 1), indices);
    return {.lower = detail::GatherLanes(values, lanes, std::make_index_sequence<kWidth>{}),
            .upper = detail::GatherLanes(values.subspan(1), lanes, std::make_index_sequence<kWidth>{})};
#endif
}

/** @brief Rounds @p count up to a whole number of vector lanes. */
constexpr size_t PadToWidth(size_t count) noexcept
{
    return ((count + kWidth - 1) / kWidth) * kWidth;
}

/**
 * @brief Returns the kWidth lanes beginning at @p offset as a fixed-extent span.
 *
 * Fixing the extent lets the Load() and Store() overloads below verify at compile time that they
 * are given exactly one vector's worth of lanes, which is what makes the callers safe without any
 * raw pointer arithmetic.
 *
 * @tparam T `float` or `const float`.
 */
template <typename T>
    requires std::is_same_v<std::remove_const_t<T>, float>
constexpr std::span<T, kWidth> LanesAt(std::span<T> data, size_t offset) noexcept SFFDN_NONBLOCKING
{
    return data.subspan(offset).template first<kWidth>();
}

/** @brief Loads one vector from exactly kWidth contiguous lanes. */
inline Vec Load(std::span<const float, kWidth> lanes) noexcept SFFDN_NONBLOCKING
{
    return Load(lanes.data());
}

/** @brief Stores one vector into exactly kWidth contiguous lanes. */
inline void Store(std::span<float, kWidth> lanes, Vec v) noexcept SFFDN_NONBLOCKING
{
    Store(lanes.data(), v);
}

} // namespace sfFDN::simd
