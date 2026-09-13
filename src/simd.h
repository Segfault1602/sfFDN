// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

// Define SFFDN_SIMD_FORCE_SCALAR to compile the portable fallback on any target. This exists so
// the scalar kernels stay testable against the vector kernels on a single machine.
#ifndef SFFDN_SIMD_FORCE_SCALAR
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SFFDN_SIMD_NEON 1
#elifdef __AVX__
#include <immintrin.h>
#define SFFDN_SIMD_AVX 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define SFFDN_SIMD_SSE 1
#endif
#endif

#if defined(SFFDN_SIMD_NEON) || defined(SFFDN_SIMD_AVX) || defined(SFFDN_SIMD_SSE)
#define SFFDN_HAS_SIMD 1
#endif

namespace sfFDN::simd
{

/**
 * @brief A single-precision vector backed by NEON, AVX, SSE, or a scalar fallback.
 *
 * The abstraction is intentionally minimal: only the operations required by the DSP kernels in
 * this library are provided. Every operation is branch-free and allocation-free so that callers
 * remain real-time safe.
 */
#ifdef SFFDN_SIMD_NEON

inline constexpr size_t kWidth = 4;
using Vec = float32x4_t;
using IntVec = int32x4_t;

inline Vec Load(const float* p) noexcept SFFDN_NONBLOCKING
{
    return vld1q_f32(p);
}

inline void Store(float* p, Vec v) noexcept SFFDN_NONBLOCKING
{
    vst1q_f32(p, v);
}

inline Vec Splat(float x) noexcept SFFDN_NONBLOCKING
{
    return vdupq_n_f32(x);
}

inline Vec Zero() noexcept SFFDN_NONBLOCKING
{
    return vdupq_n_f32(0.f);
}

inline Vec Add(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return vaddq_f32(a, b);
}

inline Vec Sub(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return vsubq_f32(a, b);
}

inline Vec Mul(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return vmulq_f32(a, b);
}

/** @brief Returns a * b + c. */
inline Vec MulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return vfmaq_f32(c, a, b);
}

/** @brief Returns c - a * b. */
inline Vec NegMulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return vfmsq_f32(c, a, b);
}

inline Vec Floor(Vec x) noexcept SFFDN_NONBLOCKING
{
    return vrndmq_f32(x);
}

inline IntVec ToInt(Vec x) noexcept SFFDN_NONBLOCKING
{
    return vcvtq_s32_f32(x);
}

inline Vec ToFloat(IntVec x) noexcept SFFDN_NONBLOCKING
{
    return vcvtq_f32_s32(x);
}

inline IntVec Min(IntVec x, int32_t maximum) noexcept SFFDN_NONBLOCKING
{
    return vminq_s32(x, vdupq_n_s32(maximum));
}

inline Vec Gather(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
    std::array<float, kWidth> gathered{};
    gathered[0] = values[static_cast<size_t>(vgetq_lane_s32(indices, 0))];
    gathered[1] = values[static_cast<size_t>(vgetq_lane_s32(indices, 1))];
    gathered[2] = values[static_cast<size_t>(vgetq_lane_s32(indices, 2))];
    gathered[3] = values[static_cast<size_t>(vgetq_lane_s32(indices, 3))];
    return Load(gathered.data());
}

#elifdef SFFDN_SIMD_AVX

inline constexpr size_t kWidth = 8;
using Vec = __m256;
using IntVec = __m256i;

inline Vec Load(const float* p) noexcept SFFDN_NONBLOCKING
{
    return _mm256_loadu_ps(p);
}

inline void Store(float* p, Vec v) noexcept SFFDN_NONBLOCKING
{
    _mm256_storeu_ps(p, v);
}

inline Vec Splat(float x) noexcept SFFDN_NONBLOCKING
{
    return _mm256_set1_ps(x);
}

inline Vec Zero() noexcept SFFDN_NONBLOCKING
{
    return _mm256_setzero_ps();
}

inline Vec Add(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm256_add_ps(a, b);
}

inline Vec Sub(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm256_sub_ps(a, b);
}

inline Vec Mul(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm256_mul_ps(a, b);
}

inline Vec MulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

inline Vec NegMulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

inline Vec Floor(Vec x) noexcept SFFDN_NONBLOCKING
{
    return _mm256_floor_ps(x);
}

inline IntVec ToInt(Vec x) noexcept SFFDN_NONBLOCKING
{
    return _mm256_cvttps_epi32(x);
}

inline Vec ToFloat(IntVec x) noexcept SFFDN_NONBLOCKING
{
    return _mm256_cvtepi32_ps(x);
}

inline IntVec Min(IntVec x, int32_t maximum) noexcept SFFDN_NONBLOCKING
{
#ifdef __AVX2__
    return _mm256_min_epi32(x, _mm256_set1_epi32(maximum));
#else
    return ToInt(_mm256_min_ps(ToFloat(x), _mm256_set1_ps(static_cast<float>(maximum))));
#endif
}

inline Vec Gather(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
#ifdef __AVX2__
    return _mm256_i32gather_ps(values.data(), indices, sizeof(float));
#else
    const auto lanes = std::bit_cast<std::array<int32_t, kWidth>>(indices);
    return _mm256_setr_ps(values[static_cast<size_t>(lanes[0])],
                          values[static_cast<size_t>(lanes[1])],
                          values[static_cast<size_t>(lanes[2])],
                          values[static_cast<size_t>(lanes[3])],
                          values[static_cast<size_t>(lanes[4])],
                          values[static_cast<size_t>(lanes[5])],
                          values[static_cast<size_t>(lanes[6])],
                          values[static_cast<size_t>(lanes[7])]);
#endif
}

#elif defined(SFFDN_SIMD_SSE)

inline constexpr size_t kWidth = 4;
using Vec = __m128;
using IntVec = __m128i;

inline Vec Load(const float* p) noexcept SFFDN_NONBLOCKING
{
    return _mm_loadu_ps(p);
}

inline void Store(float* p, Vec v) noexcept SFFDN_NONBLOCKING
{
    _mm_storeu_ps(p, v);
}

inline Vec Splat(float x) noexcept SFFDN_NONBLOCKING
{
    return _mm_set1_ps(x);
}

inline Vec Zero() noexcept SFFDN_NONBLOCKING
{
    return _mm_setzero_ps();
}

inline Vec Add(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm_add_ps(a, b);
}

inline Vec Sub(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm_sub_ps(a, b);
}

inline Vec Mul(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return _mm_mul_ps(a, b);
}

inline Vec MulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}

inline Vec NegMulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
}

inline Vec Floor(Vec x) noexcept SFFDN_NONBLOCKING
{
    const Vec truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
    const Vec correction = _mm_and_ps(_mm_cmplt_ps(x, truncated), _mm_set1_ps(1.f));
    return _mm_sub_ps(truncated, correction);
}

inline IntVec ToInt(Vec x) noexcept SFFDN_NONBLOCKING
{
    return _mm_cvttps_epi32(x);
}

inline Vec ToFloat(IntVec x) noexcept SFFDN_NONBLOCKING
{
    return _mm_cvtepi32_ps(x);
}

inline IntVec Min(IntVec x, int32_t maximum) noexcept SFFDN_NONBLOCKING
{
    const IntVec maximum_vector = _mm_set1_epi32(maximum);
    const IntVec overflow = _mm_cmpgt_epi32(x, maximum_vector);
    return _mm_or_si128(_mm_and_si128(overflow, maximum_vector), _mm_andnot_si128(overflow, x));
}

inline Vec Gather(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
    const auto lanes = std::bit_cast<std::array<int32_t, kWidth>>(indices);
    return _mm_setr_ps(values[static_cast<size_t>(lanes[0])], values[static_cast<size_t>(lanes[1])],
                       values[static_cast<size_t>(lanes[2])], values[static_cast<size_t>(lanes[3])]);
}

#else

inline constexpr size_t kWidth = 4;

struct Vec
{
    float v[kWidth];
};
using IntVec = std::array<int32_t, kWidth>;

inline Vec Load(const float* p) noexcept SFFDN_NONBLOCKING
{
    return Vec{{p[0], p[1], p[2], p[3]}};
}

inline void Store(float* p, Vec v) noexcept SFFDN_NONBLOCKING
{
    p[0] = v.v[0];
    p[1] = v.v[1];
    p[2] = v.v[2];
    p[3] = v.v[3];
}

inline Vec Splat(float x) noexcept SFFDN_NONBLOCKING
{
    return Vec{{x, x, x, x}};
}

inline Vec Zero() noexcept SFFDN_NONBLOCKING
{
    return Vec{{0.f, 0.f, 0.f, 0.f}};
}

inline Vec Add(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return Vec{{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3]}};
}

inline Vec Sub(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return Vec{{a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2], a.v[3] - b.v[3]}};
}

inline Vec Mul(Vec a, Vec b) noexcept SFFDN_NONBLOCKING
{
    return Vec{{a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3]}};
}

inline Vec MulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return Vec{{(a.v[0] * b.v[0]) + c.v[0],
                (a.v[1] * b.v[1]) + c.v[1],
                (a.v[2] * b.v[2]) + c.v[2],
                (a.v[3] * b.v[3]) + c.v[3]}};
}

inline Vec NegMulAdd(Vec a, Vec b, Vec c) noexcept SFFDN_NONBLOCKING
{
    return Vec{{c.v[0] - (a.v[0] * b.v[0]),
                c.v[1] - (a.v[1] * b.v[1]),
                c.v[2] - (a.v[2] * b.v[2]),
                c.v[3] - (a.v[3] * b.v[3])}};
}

inline Vec Floor(Vec x) noexcept SFFDN_NONBLOCKING
{
    return Vec{{std::floor(x.v[0]), std::floor(x.v[1]), std::floor(x.v[2]), std::floor(x.v[3])}};
}

inline IntVec ToInt(Vec x) noexcept SFFDN_NONBLOCKING
{
    return {static_cast<int32_t>(x.v[0]), static_cast<int32_t>(x.v[1]), static_cast<int32_t>(x.v[2]),
            static_cast<int32_t>(x.v[3])};
}

inline Vec ToFloat(IntVec x) noexcept SFFDN_NONBLOCKING
{
    return Vec{{static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]), static_cast<float>(x[3])}};
}

inline IntVec Min(IntVec x, int32_t maximum) noexcept SFFDN_NONBLOCKING
{
    for (int32_t& lane : x)
    {
        lane = std::min(lane, maximum);
    }
    return x;
}

inline Vec Gather(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
    return Vec{{values[static_cast<size_t>(indices[0])], values[static_cast<size_t>(indices[1])],
                values[static_cast<size_t>(indices[2])], values[static_cast<size_t>(indices[3])]}};
}

#endif

struct AdjacentGather
{
    Vec lower;
    Vec upper;
};

inline AdjacentGather GatherAdjacent(std::span<const float> values, IntVec indices) noexcept SFFDN_NONBLOCKING
{
#ifdef SFFDN_SIMD_NEON
    const float32x2_t pair0 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(indices, 0))]);
    const float32x2_t pair1 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(indices, 1))]);
    const float32x2_t pair2 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(indices, 2))]);
    const float32x2_t pair3 = vld1_f32(&values[static_cast<size_t>(vgetq_lane_s32(indices, 3))]);
    const float32x4_t low = vcombine_f32(pair0, pair1);
    const float32x4_t high = vcombine_f32(pair2, pair3);
    return {vuzp1q_f32(low, high), vuzp2q_f32(low, high)};
#else
    return {Gather(values, indices), Gather(values.subspan(1), indices)};
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
