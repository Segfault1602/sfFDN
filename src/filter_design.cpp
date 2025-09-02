#include "sffdn/filter_design.h"
#include "filter_design_internal.h"

#include "pch.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>

namespace
{
template <typename T>
T db2mag(T x)
{
    return std::pow(static_cast<T>(10), x / static_cast<T>(20));
}

float RT602Slope(float t60, float sr)
{
    return -60.f / (t60 * sr);
}

template <typename T>
void ToDb(std::span<const T> x, std::span<T> out)
{
    for (size_t i = 0; i < x.size(); ++i)
    {
        out[i] = 20.0 * std::log10(x[i]);
    }
}

template <typename T>
void freqz(std::span<const T> b, std::span<const T> a, std::span<std::complex<T>> w, std::span<T> result)
{
    if (b.size() > 3 || a.size() > 3)
    {
        throw std::runtime_error("Only tested for first-order filters (b.size() <= 3 and a.size() <= 3)");
    }

    assert(result.size() == (w.size()));

    std::vector<std::complex<T>> num(w.size());
    std::vector<std::complex<T>> den(w.size());

    sfFDN::Polyval<T>(b, w, num);
    sfFDN::Polyval<T>(a, w, den);

    for (auto [n_, d_, h_] : std::views::zip(num, den, result))
    {
        T a = n_.real();
        T b = n_.imag();
        T c = d_.real();
        T d = d_.imag();

        T c2 = c * c;
        T d2 = d * d;

        T x = (a * c + b * d) / (c2 + d2);
        T y = (a * d - b * c) / (c2 + d2);

        h_ = std::sqrt((x * x) + (y * y));
    }
}

template <typename T, size_t kNBands, size_t kNFreqs>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> InteractionMatrix(std::span<const T> G, T kGW, std::span<const T> wg,
                                                                   std::span<const T> wc, std::span<const T> bw)
{
    Eigen::Matrix<T, kNBands, kNFreqs> leak = Eigen::Matrix<T, kNBands, kNFreqs>::Zero();

    std::array<T, kNBands> Gdb{};
    ToDb<T>(G, Gdb);

    T gdb_abs_sum =
        std::accumulate(Gdb.begin(), Gdb.end(), static_cast<T>(0), [](T sum, T val) { return sum + std::abs(val); });
    if (gdb_abs_sum <= 1e-15)
    {
        for (int i = 0; i < kNBands; ++i)
        {
            leak(i, i * 2) = 1;
        }
        return leak;
    }

    std::array<T, kNBands> Gw{};
    std::ranges::transform(Gdb, Gw.begin(), [kGW](T val) -> T { return db2mag(kGW * val); });

    std::array<std::complex<T>, kNFreqs> dig_w_arr{};
    for (auto [w, f] : std::views::zip(dig_w_arr, wc))
    {
        w = std::exp(std::complex<T>(0.0, 1.0) * f);
    }

    for (auto i = 0; i < kNBands; ++i)
    {
        std::array<T, 6> sos = sfFDN::Pareq(G[i], Gw[i], wg[i], bw[i]);
        auto sos_span = std::span<T>(sos);
        auto num = sos_span.first(3);
        auto den = sos_span.last(3);
        std::array<T, kNFreqs> H{};
        freqz<T>(num, den, dig_w_arr, H);

        for (auto j = 0; j < kNFreqs; ++j)
        {
            leak(i, j) = (20.0 * std::log10(H[j])) / Gdb[i];
        }
    }

    return leak;
}

template <typename T, size_t kNBands>
std::vector<T> aceq(std::span<const T> diff_mag, std::span<const T> freqs, T sr)
{
    if (diff_mag.size() != kNBands || freqs.size() != kNBands)
    {
        throw std::runtime_error("diff_mag and freqs must have size " + std::to_string(kNBands));
    }

    constexpr size_t kNFreqs = (kNBands * 2) - 1;
    constexpr T kGW = 0.3; // Gain factor at bandwidth

    // array of center frequencies + intermediate frequencies
    std::array<T, kNFreqs> fc2 = {0};
    for (auto i = 0; i < freqs.size(); ++i)
    {
        fc2.at(i * 2) = freqs[i];
    }

    for (auto i = 1; i < fc2.size(); i += 2)
    {
        fc2.at(i) = std::sqrt(fc2.at(i - 1) * fc2.at(i + 1));
    }

    // Command gain frequencies in radians
    std::array<T, kNBands> wg = {0.0f};
    for (auto [w, f] : std::views::zip(wg, freqs))
    {
        w = 2 * std::numbers::pi_v<T> * f / sr;
    }

    // Center frequencies in radian for iterative design
    std::array<T, kNFreqs> wc = {0.0f};
    for (auto [w, f] : std::views::zip(wc, fc2))
    {
        w = 2 * std::numbers::pi_v<T> * f / sr;
    }

    std::array<T, kNBands> bw = {0.0f};
    for (auto [b, w] : std::views::zip(bw, wg))
    {
        b = 1.5 * w;
    }
    // Extra adjustment
    if constexpr (kNBands == 10)
    {
        bw[7] *= 0.93;
        bw[8] *= 0.78;
        bw[9] = 0.76 * wg[9];
    }

    std::array<T, kNBands> G{};
    G.fill(std::pow(10.0, kNFreqs / 20.0));

    Eigen::Matrix<T, kNBands, kNFreqs> leak = InteractionMatrix<T, kNBands, kNFreqs>(G, kGW, wg, wc, bw);

    Eigen::Map<const Eigen::Array<T, kNBands, 1>> Gdb(diff_mag.data(), diff_mag.size());

    Eigen::Vector<T, kNFreqs> Gdb2 = Eigen::Vector<T, kNFreqs>::Zero();
    Gdb2(Eigen::seq(0, kNFreqs - 1, 2)) = Gdb;
    Gdb2(Eigen::seq(1, kNFreqs - 1, 2)) =
        (Gdb2(Eigen::seq(0, kNFreqs - 3, 2)) + Gdb2(Eigen::seq(2, kNFreqs - 1, 2))) / 2;

    // Solve least squares optmization problem
    Eigen::Vector<T, kNBands> solution = (leak * leak.transpose()).ldlt().solve(leak * Gdb2);

    std::array<T, kNBands> goptdb{};
    Eigen::Map<Eigen::Array<T, kNBands, 1>> goptdb_map(goptdb.data());
    goptdb_map = Eigen::pow(10.0, solution.array() / 20);

    Eigen::Array<T, kNBands, 1> gwopt = Eigen::pow(10.0, kGW * solution.array() / 20.0);

    Eigen::Matrix<T, kNBands, kNFreqs> leak2 = InteractionMatrix<T, kNBands, kNFreqs>(goptdb, kGW, wg, wc, bw);
    Eigen::Vector<T, kNBands> solution2 = (leak2 * leak2.transpose()).ldlt().solve(leak2 * Gdb2);

    goptdb_map = Eigen::pow(10.0, solution2.array() / 20);
    gwopt = Eigen::pow(10.0, kGW * solution2.array() / 20);

    std::vector<T> sos;
    for (auto i = 0; i < kNBands; ++i)
    {
        std::array<T, 6> coeffs = sfFDN::Pareq(goptdb[i], gwopt[i], wg.at(i), bw.at(i));
        sos.insert(sos.end(), coeffs.begin(), coeffs.end());
    }

    return sos;
}

template <typename T>
std::vector<T> GetTwoFilter_impl(std::span<const T> t60s, T delay, T sr, T shelf_cutoff)
{
    constexpr size_t kNBands = 10;

    if (t60s.size() != kNBands)
    {
        throw std::runtime_error("t60s must have size " + std::to_string(kNBands));
    }

    std::vector<T> freqs(kNBands, 0.0);
    constexpr T kUpperLimit = 16000.0f;
    for (auto i = 0; i < kNBands; ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0, static_cast<T>(kNBands - 1 - i));
    }

    std::vector<T> gains(kNBands, 0.0f);
    for (auto i = 0; i < kNBands; ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], delay / sr);
        gains[i] = 20.0 * std::log10(gains[i]);
    }

    std::vector<T> linear_gains(gains.size(), 0.0);
    for (auto i = 0; i < gains.size(); ++i)
    {
        linear_gains[i] = db2mag(gains[i]);
    }

    // Build first-order low shelf filter
    T gain_low = linear_gains[0];
    T gain_high = linear_gains[linear_gains.size() - 1];

    std::array<T, 4> shelf_sos = sfFDN::LowShelf(shelf_cutoff, sr, gain_low, gain_high);
    std::span shelf_sos_span{shelf_sos};

    std::array<T, 3> b_coeffs = {shelf_sos[0] / shelf_sos[2], shelf_sos[1] / shelf_sos[2], 0.0f};
    std::array<T, 3> a_coeffs = {1.0f, shelf_sos[3] / shelf_sos[2], 0.0f};

    std::vector<std::complex<T>> dig_w(kNBands);
    for (size_t i = 0; i < kNBands; ++i)
    {
        dig_w[i] = std::exp(std::complex<T>(0.0, 1.0) * freqs[i] * (-2 * std::numbers::pi_v<T> / sr));
    }

    std::array<T, kNBands> Hshelf{};
    freqz<T>(b_coeffs, a_coeffs, dig_w, Hshelf);

    std::vector<T> diff_mag(freqs.size(), 0.0f);
    for (auto i = 0; i < freqs.size(); ++i)
    {
        diff_mag[i] = gains[i] - 20 * std::log10(Hshelf[i]);
    }

    std::vector<T> sos_T;
    if (kNBands == 10) // octave bands
    {
        sos_T = aceq<T, kNBands>(diff_mag, freqs, sr);
    }

    assert(sos_T.size() == kNBands * 6);

    std::vector<T> sos(sos_T.size() + 6, 0.0f);

    // Copy the low shelf filter coefficients
    sos[0] = shelf_sos[0] / shelf_sos[2];
    sos[1] = shelf_sos[1] / shelf_sos[2];
    sos[2] = 0.0f;
    sos[3] = 1.f;
    sos[4] = shelf_sos[3] / shelf_sos[2];
    sos[5] = 0.0f;

    for (auto i = 0; i < sos_T.size(); ++i)
    {
        sos[i + 6] = sos_T[i];
    }

    return sos;
}
} // namespace

namespace sfFDN
{

// From: https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/auxiliary/onePoleAbsorption.m
// Based on Jot, J. M., & Chaigne, A. (1991). Digital delay networks for designing artificial reverberators (pp. 1-12).
// Presented at the Proc. Audio Eng. Soc. Conv., Paris, France.
void GetOnePoleAbsorption(float t60_dc, float t60_ny, float sr, float delay, float& b, float& a)
{
    const float h_dc = db2mag(delay * RT602Slope(t60_dc, sr));
    const float h_ny = db2mag(delay * RT602Slope(t60_ny, sr));

    const float r = h_dc / h_ny;
    a = (1 - r) / (1 + r);
    b = (1 - a) * h_ny;
}

std::vector<double> GetTwoFilter_d(std::span<const double> t60s, double delay, double sr, double shelf_cutoff)
{
    return GetTwoFilter_impl<double>(t60s, delay, sr, shelf_cutoff);
}

std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff)
{
    // The coefficients are computed in double precision, otherwise there is a significant loss of precision and the
    // filter is not as accurate as it could be.
    std::vector<double> t60s_d(t60s.begin(), t60s.end());
    std::vector<double> sos = GetTwoFilter_d(t60s_d, static_cast<double>(delay), static_cast<double>(sr), shelf_cutoff);

    std::vector<float> sos_f(sos.begin(), sos.end());

    return sos_f;
}

std::vector<float> DesignGraphicEQ(std::span<const float> mag, std::span<const float> freqs, float sr)
{
    std::vector<double> mag_d(mag.begin(), mag.end());
    std::vector<double> freqs_d(freqs.begin(), freqs.end());
    std::vector<double> sos = aceq<double, 10>(mag_d, freqs_d, static_cast<double>(sr));
    std::vector<float> sos_f(sos.begin(), sos.end());
    return sos_f;
}

} // namespace sfFDN