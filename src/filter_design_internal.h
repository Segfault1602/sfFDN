#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>


namespace sfFDN
{
template <typename T>
std::array<T, 4> LowShelf(T wc, T sr, T gain_low, T gain_high);

template <typename T>
std::array<T, 6> LowShelfRBJ(T wc, T db_gain, T Q);

template <typename T>
std::array<T, 6> HighShelfRBJ(T wc, T db_gain, T Q);

template <typename T>
std::array<T, 6> Pareq(T g, T gb, T w0, T b);

template <typename T>
void Polyval(const std::span<const T> p, const std::span<std::complex<T>> x, std::span<std::complex<T>> result);

std::vector<double> GetTwoFilter_d(std::span<const double> t60s, double delay, double sr, double shelf_cutoff = 8000.0);

} // namespace sfFDN

template <typename T>
std::array<T, 4> sfFDN::LowShelf(T wc, T sr, T gain_low, T gain_high)
{
    const T wH = 2 * std::numbers::pi * wc / sr;
    const T g = gain_low / gain_high;
    const T g_sqrt = std::sqrt(g);

    T ah0 = std::tan(wH * 0.5) + g_sqrt;
    T ah1 = std::tan(wH * 0.5) - g_sqrt;
    T bh0 = (g * std::tan(wH * 0.5)) + g_sqrt;
    T bh1 = (g * std::tan(wH * 0.5)) - g_sqrt;

    std::array<T, 4> sos = {gain_high * bh0, gain_high * bh1, ah0, ah1};
    return sos;
}

template <typename T>
std::array<T, 6> sfFDN::LowShelfRBJ(T wc, T db_gain, T Q)
{
    const T A = std::pow(10.0, db_gain / 40.0);
    const T w0 = 2 * std::numbers::pi * wc;
    const T alpha = std::sin(w0) / (2.0 * Q);
    const T cos_w0 = std::cos(w0);
    const T TwoSqrtA = (2 * std::sqrt(A) * alpha);

    const T b0 = A * ((A + 1) - (A - 1) * cos_w0 + TwoSqrtA);
    const T b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0);
    const T b2 = A * ((A + 1) - (A - 1) * cos_w0 - TwoSqrtA);

    const T a0 = (A + 1) + ((A - 1) * cos_w0) + TwoSqrtA;
    const T a1 = -2 * ((A - 1) + (A + 1) * cos_w0);
    const T a2 = (A + 1) + ((A - 1) * cos_w0) - TwoSqrtA;

    return {b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0};
}

template <typename T>
std::array<T, 6> sfFDN::HighShelfRBJ(T wc, T db_gain, T Q)
{
    const T A = std::pow(10.0, db_gain / 40.0);
    const T w0 = 2 * std::numbers::pi * wc;
    const T alpha = std::sin(w0) / (2.0 * Q);
    const T cos_w0 = std::cos(w0);
    const T TwoSqrtA = (2 * std::sqrt(A) * alpha);

    const T b0 = A * ((A + 1) + (A - 1) * cos_w0 + TwoSqrtA);
    const T b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0);
    const T b2 = A * ((A + 1) + (A - 1) * cos_w0 - TwoSqrtA);

    const T a0 = (A + 1) - ((A - 1) * cos_w0) + TwoSqrtA;
    const T a1 = 2 * ((A - 1) - (A + 1) * cos_w0);
    const T a2 = (A + 1) - ((A - 1) * cos_w0) - TwoSqrtA;

    return {b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0};
}

template <typename T>
std::array<T, 6> sfFDN::Pareq(T g, T gb, T w0, T b)
{
    T beta = 0.f;
    if (g == 1.f)
    {
        beta = std::tan(b / 2);
    }
    else
    {
        beta = std::sqrt(std::abs((gb * gb) - 1.f) / std::abs((g * g) - (gb * gb))) * std::tan(b / 2);
    }

    const T b0 = (1 + g * beta) / (1 + beta);
    const T b1 = (-2 * std::cos(w0)) / (1 + beta);
    const T b2 = (1 - g * beta) / (1 + beta);

    const T a0 = 1.0;
    const T a1 = -2 * std::cos(w0) / (1 + beta);
    const T a2 = (1 - beta) / (1 + beta);

    return {b0, b1, b2, a0, a1, a2};
}

template <typename T>
void sfFDN::Polyval(const std::span<const T> p, const std::span<std::complex<T>> x, std::span<std::complex<T>> result)
{
    if (p.size() < 1)
    {
        return;
    }

    assert(x.size() == result.size());

    std::ranges::fill(result, p[0]);

    for (auto i = 1; i < p.size(); ++i)
    {
        for (auto [rval, xval] : std::views::zip(result, x))
        {
            rval = xval * rval + p[i];
        }
    }
}
