#include "sffdn/filter_design.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace
{
float db2mag(float x)
{
    return pow(10, x / 20.f);
}

float RT602Slope(float t60, float sr)
{
    return -60 / (t60 * sr);
}

std::vector<double> logspace(double start, double stop, size_t num)
{
    std::vector<double> result(num);
    if (num == 0)
        return result;

    double step = (stop - start) / (num - 1);

    Eigen::Map<Eigen::ArrayXd> result_map(result.data(), num);

    result_map = Eigen::ArrayXd::LinSpaced(num, start, stop);
    result_map = Eigen::pow(10, result_map);

    return result;
}

std::vector<size_t> find_indices(std::span<const double> w, std::span<const double> freqs)
{
    std::vector<size_t> ind(freqs.size(), 0);
    size_t current_band_idx = 0;
    size_t current_freq_idx = 0;
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        while (w[current_freq_idx] < freqs[i])
        {
            current_freq_idx++;
        }

        double diff1 = std::abs(w[current_freq_idx] - freqs[i]);
        double diff2 = std::abs(w[current_freq_idx - 1] - freqs[i]);

        if (diff1 < diff2)
        {
            ind[i] = current_freq_idx;
        }
        else
        {
            ind[i] = current_freq_idx - 1;
        }

        if (current_freq_idx >= w.size() - 1)
        {
            // This should not happen as long as the sample rate is high enough
            std::cerr << "Warning: Frequency index out of bounds. Using the last frequency index." << std::endl;
            ind[i] = w.size() - 1;
        }
    }
    return ind;
}

std::vector<double> interp1(std::span<const double> x, std::span<const double> y, std::span<const double> xi)
{
    std::vector<double> yi(xi.size(), 0.0f);

    if (x.size() != y.size())
    {
        throw std::runtime_error("x and y must have the same size");
    }

    if (x.size() < 2)
    {
        throw std::runtime_error("x and y must have at least two points for interpolation");
    }

    for (size_t i = 0; i < xi.size(); ++i)
    {
        // Find the interval in which xi[i] lies
        auto it = std::lower_bound(x.begin(), x.end(), xi[i]);
        if (it == x.end())
        {
            // If xi[i] is greater than the last element in x, use the last value
            yi[i] = y.back();
            continue;
        }

        size_t idx = std::distance(x.begin(), it);

        if (idx == 0)
        {
            // If xi[i] is less than the first element in x, use the first value
            yi[i] = y.front();
        }
        else
        {
            double slope = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1]);
            yi[i] = y[idx - 1] + slope * (xi[i] - x[idx - 1]);
        }
    }

    return yi;
}

std::array<double, 4> LowShelf(double wc, double sr, double gain_low, double gain_high)
{
    const double wH = 2 * std::numbers::pi * wc / sr;
    const double g = gain_low / gain_high;
    const double g_sqrt = std::sqrt(g);

    double ah0 = std::tan(wH / 2.f) + g_sqrt;
    double ah1 = std::tan(wH / 2.f) - g_sqrt;
    double bh0 = g * std::tan(wH / 2.f) + g_sqrt;
    double bh1 = g * std::tan(wH / 2.f) - g_sqrt;

    std::array<double, 4> sos = {gain_high * bh0, gain_high * bh1, ah0, ah1};
    // Normalize the coefficients
    for (size_t i = 0; i < sos.size(); ++i)
    {
        sos[i] /= ah0;
    }
    return sos;
}

Eigen::ArrayXcd Polyval(const Eigen::ArrayXd& p, const Eigen::ArrayXcd& x)
{
    Eigen::ArrayXcd result = Eigen::ArrayXcd::Zero(x.size());
    result += p[0];

    for (size_t i = 1; i < p.size(); ++i)
    {
        result = x * result + p[i];
    }

    return result;
}

std::vector<double> freqz(std::span<const double> b, std::span<const double> a, std::span<const double> w,
                          double sr = 0.f)
{
    if (b.size() > 3 || a.size() > 3)
    {
        throw std::runtime_error("Only tested for first-order filters (b.size() == 2 and a.size() == 2)");
    }

    Eigen::Map<const Eigen::ArrayXd> w_map(w.data(), w.size());
    Eigen::ArrayXcd dig_w(w.size());
    // if sample rate is specified, convert to rad/sample
    if (sr != 0.0f)
    {
        dig_w = Eigen::exp(std::complex(0.0, 1.0) * w_map * (-2.0f * std::numbers::pi_v<double> / sr));
    }
    else
    {
        dig_w = Eigen::exp(std::complex(0.0, 1.0) * w_map);
    }

    Eigen::Map<const Eigen::ArrayXd> b_map(b.data(), b.size());
    auto num = Polyval(b_map, dig_w);

    Eigen::Map<const Eigen::ArrayXd> a_map(a.data(), a.size());
    auto den = Polyval(a_map, dig_w);

    auto h_complex = num / den;

    std::vector<double> h(w.size(), 0.0f);
    Eigen::Map<Eigen::ArrayXd> h_map(h.data(), h.size());
    h_map = h_complex.abs();

    return h;
}

std::array<double, 6> Pareq(double g, double gb, double w0, double b)
{
    double beta = 0.f;
    if (g == 1.f)
    {
        beta = std::tan(b / 2);
    }
    else
    {
        beta = std::sqrt(std::abs(gb * gb - 1.f) / std::abs(g * g - gb * gb)) * std::tan(b / 2);
    }

    const double b0 = (1 + g * beta) / (1 + beta);
    const double b1 = (-2 * std::cos(w0)) / (1 + beta);
    const double b2 = (1 - g * beta) / (1 + beta);

    const double a0 = 1.f;
    const double a1 = -2 * std::cos(w0) / (1 + beta);
    const double a2 = (1 - beta) / (1 + beta);

    return {b0, b1, b2, a0, a1, a2};
}

Eigen::MatrixXd InteractionMatrix(std::span<const double> G, double kGW, std::span<const double> wg,
                                  std::span<const double> wc, std::span<const double> bw, double sr)
{
    const size_t kM = wg.size();
    const size_t kN = wc.size();

    Eigen::MatrixXd leak = Eigen::MatrixXd::Zero(kM, kN);

    Eigen::Map<const Eigen::ArrayXd> G_map(G.data(), G.size());
    Eigen::ArrayXd Gdb = 20.f * G_map.log10();
    Eigen::ArrayXd Gw = kGW * Gdb;
    Gw = Eigen::pow(10.f, Gw / 20.f);

    if (Gdb.sum() == 0.0f)
    {
        return leak;
    }

    for (size_t i = 0; i < kM; ++i)
    {
        std::array<double, 6> sos = Pareq(G[i], Gw[i], wg[i], bw[i]);
        auto num = std::span<double>(sos.data(), 3);
        auto den = std::span<double>(sos.data() + 3, 3);
        std::vector<double> H = freqz(num, den, wc);

        Eigen::Map<Eigen::ArrayXd> H_map(H.data(), H.size());
        H_map = 20.f * H_map.log10();

        Eigen::ArrayXd Gain = H_map / Gdb[i];

        leak.row(i) = Gain;
    }

    return leak;
}

std::vector<double> aceq_d(std::span<const double> diff_mag, std::span<const double> freqs, double sr)
{
    constexpr size_t kNBands = 10;
    constexpr size_t kNumF = 19;
    constexpr double kGW = 0.3; // Gain factor at bandwidth

    if (diff_mag.size() != kNBands || freqs.size() != kNBands)
    {
        throw std::runtime_error("diff_mag and freqs must have size " + std::to_string(kNBands));
    }

    // array of center frequencies + intermediate frequencies
    std::array<double, kNumF> fc2 = {0};
    for (size_t i = 0; i < fc2.size(); i += 2)
    {
        fc2[i] = freqs[i / 2];
    }
    for (size_t i = 1; i < fc2.size(); i += 2)
    {
        fc2[i] = std::sqrt(fc2[i - 1] * fc2[i + 1]);
    }

    // Command gain frequencies in radians
    std::array<double, kNBands> wg = {0.0f};
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        wg[i] = 2 * std::numbers::pi_v<double> * freqs[i] / sr;
    }

    // Center frequencies in radian for iterative design
    std::array<double, kNumF> wc = {0.0f};
    for (size_t i = 0; i < fc2.size(); ++i)
    {
        wc[i] = 2 * std::numbers::pi_v<double> * fc2[i] / sr;
    }

    std::array<double, kNBands> bw = {0.0f};
    for (size_t i = 0; i < wg.size(); ++i)
    {
        bw[i] = 1.5f * wg[i];
    }
    // Extra adjustment
    bw[7] *= 0.93f;
    bw[8] *= 0.78f;
    bw[9] = 0.76f * wg[9]; // todo: check if this is a bug in the original code

    std::array<double, kNBands> G;
    std::fill(G.begin(), G.end(), std::pow(10.f, kNumF / 20.f));

    auto leak = InteractionMatrix(G, kGW, wg, wc, bw, sr);

    Eigen::Map<const Eigen::ArrayXd> Gdb(diff_mag.data(), diff_mag.size());

    Eigen::VectorXd Gdb2 = Eigen::VectorXd::Zero(kNumF);
    Gdb2(Eigen::seq(0, kNumF - 1, 2)) = Gdb;
    Gdb2(Eigen::seq(1, kNumF - 1, 2)) = (Gdb2(Eigen::seq(0, kNumF - 3, 2)) + Gdb2(Eigen::seq(2, kNumF - 1, 2))) / 2;

    // Transpose the leak matrix for the SVD solve
    leak.transposeInPlace();
    Eigen::VectorXd solution = leak.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Gdb2);
    Eigen::ArrayXd goptdb = Eigen::pow(10.f, solution.array() / 20.f);
    Eigen::ArrayXd gwopt = Eigen::pow(10.f, kGW * solution.array() / 20.f);

    constexpr int kNumIteration = 1;

    for (int iter = 0; iter < kNumIteration; ++iter)
    {
        auto leak2 = InteractionMatrix(goptdb, kGW, wg, wc, bw, sr);
        leak2.transposeInPlace();
        Eigen::VectorXd solution2 = leak2.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Gdb2);

        goptdb = Eigen::pow(10.f, solution2.array() / 20.f);
        gwopt = Eigen::pow(10.f, kGW * solution2.array() / 20.f);
    }

    std::vector<double> sos;
    for (size_t i = 0; i < kNBands; ++i)
    {
        std::array<double, 6> coeffs = Pareq(goptdb[i], gwopt[i], wg[i], bw[i]);
        sos.insert(sos.end(), coeffs.begin(), coeffs.end());
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

std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff)
{
    const auto kNBands = t60s.size();
    std::vector<double> freqs(kNBands, 0.0);
    if (kNBands == 10) // octave bands
    {
        constexpr double kUpperLimit = 16000.0f;
        for (size_t i = 0; i < kNBands; ++i)
        {
            freqs[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
        }
    }
    else if (kNBands == 30 || kNBands == 31) // third octave bands
    {
        for (size_t i = 0; i < kNBands; ++i)
        {
            freqs[i] = std::pow(10.f, 3) * std::pow(2.f, static_cast<float>(-17.f + i) / 3.f);
        }
    }
    else
    {
        throw std::runtime_error(std::format("Unsupported number of bands for two_filter: {}", kNBands));
    }

    std::vector<double> gains(kNBands, 0.0f);
    for (size_t i = 0; i < kNBands; ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], delay / sr);
        gains[i] = 20.f * std::log10(gains[i]);
    }

    const size_t kNfreq = std::pow(2.f, 9);
    std::vector<double> w = logspace(std::log10(1), std::log10(sr / 2 - 1), kNfreq - 1);
    w.push_back(sr / 2); // Add Nyquist frequency

    // locate the closest frequency to each band
    std::vector<size_t> ind = find_indices(w, freqs);

    std::vector<double> target_mag = interp1(freqs, gains, w);

    std::vector<double> linear_gains(target_mag.size(), 0.0f);
    for (size_t i = 0; i < target_mag.size(); ++i)
    {
        linear_gains[i] = db2mag(target_mag[i]);
    }

    // Build first-order low shelf filter
    double gain_low = linear_gains[0];
    double gain_high = linear_gains[linear_gains.size() - 1];

    std::array<double, 4> shelf_sos = LowShelf(shelf_cutoff, sr, gain_low, gain_high);

    auto b_coeffs = std::span<double>(shelf_sos.data(), 2);
    auto a_coeffs = std::span<double>(shelf_sos.data() + 2, 2);
    std::vector<double> Hshelf = freqz(b_coeffs, a_coeffs, w, sr);

    std::vector<double> diff_mag(freqs.size(), 0.0f);
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        diff_mag[i] = target_mag[ind[i]] - 20 * std::log10(Hshelf[ind[i]]);
    }

    std::vector<double> sos_double;
    if (kNBands == 10) // octave bands
    {
        sos_double = aceq_d(diff_mag, freqs, sr);
    }

    assert(sos_double.size() == kNBands * 6);

    std::vector<float> sos(sos_double.size() + 6, 0.0f);

    // Copy the low shelf filter coefficients
    sos[0] = static_cast<float>(shelf_sos[0]);
    sos[1] = static_cast<float>(shelf_sos[1]);
    sos[2] = 0.f;
    sos[3] = static_cast<float>(shelf_sos[2]);
    sos[4] = static_cast<float>(shelf_sos[3]);
    sos[5] = 0.0f;

    for (size_t i = 0; i < sos_double.size(); ++i)
    {
        sos[i + 6] = static_cast<float>(sos_double[i]);
    }

    return sos;
}

std::vector<float> aceq(std::span<const float> mag, std::span<const float> freqs, float sr)
{
    std::vector<double> mag_d(mag.begin(), mag.end());
    std::vector<double> freqs_d(freqs.begin(), freqs.end());
    std::vector<double> sos = aceq_d(mag_d, freqs_d, static_cast<double>(sr));
    std::vector<float> sos_f(sos.begin(), sos.end());
    return sos_f;
}

} // namespace sfFDN
