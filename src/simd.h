// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"

#include <cstddef>
#include <span>
#include <type_traits>

// Define SFFDN_SIMD_FORCE_SCALAR to compile the portable fallback on any target. This exists so
// the scalar kernels stay testable against the vector kernels on a single machine.
#if !defined(SFFDN_SIMD_FORCE_SCALAR)
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SFFDN_SIMD_NEON 1
#elif defined(__AVX__)
#include <immintrin.h>
#define SFFDN_SIMD_AVX 1
#elif defined(__SSE__) || defined(HAVE_XMMINTRIN_H) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
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
#if defined(SFFDN_SIMD_NEON)

inline constexpr size_t kWidth = 4;
using Vec = float32x4_t;

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

#elif defined(SFFDN_SIMD_AVX)

inline constexpr size_t kWidth = 8;
using Vec = __m256;

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

#elif defined(SFFDN_SIMD_SSE)

inline constexpr size_t kWidth = 4;
using Vec = __m128;

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

#else

inline constexpr size_t kWidth = 4;

struct Vec
{
    float v[kWidth];
};

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

#endif

/** @brief Rounds @p count up to a whole number of vector lanes. */
inline constexpr size_t PadToWidth(size_t count) noexcept
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
