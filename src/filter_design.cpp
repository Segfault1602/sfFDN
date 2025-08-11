#include "sffdn/filter_design.h"
#include "filter_design_internal.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

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

Eigen::MatrixXd InteractionMatrix(const Eigen::ArrayXd& G, double kGW, std::span<const double> wg,
                                  std::span<const double> wc, std::span<const double> bw)
{
    // const uint32_t kM = wg.size(); // 10
    // const uint32_t kN = wc.size(); // 19

    constexpr int kM = 10;
    constexpr int kN = 19;

    Eigen::Matrix<double, kM, kN> leak = Eigen::MatrixXd::Zero(kM, kN);

    Eigen::Array<double, kM, 1> Gdb = 20 * G.log10();
    Eigen::Array<double, kM, 1> Gw = kGW * Gdb;
    Gw = Eigen::pow(10.0, Gw / 20.0);

    if (Gdb.sum() == 0.0f)
    {
        return leak;
    }

    for (auto i = 0; i < kM; ++i)
    {
        std::array<double, 6> sos = sfFDN::Pareq(G[i], Gw[i], wg[i], bw[i]);
        auto sos_span = std::span<double>(sos.data(), sos.size());
        auto num = sos_span.first(3);
        auto den = sos_span.last(3);
        std::vector<double> H = sfFDN::freqz(num, den, wc);

        Eigen::Map<Eigen::Array<double, kN, 1>> H_map(H.data(), H.size());
        H_map = 20.0 * H_map.log10();

        Eigen::Array<double, kN, 1> Gain = H_map / Gdb[i];

        leak.row(i) = Gain;
    }

    return leak;
}

} // namespace

namespace sfFDN
{
std::array<double, 4> LowShelf(double wc, double sr, double gain_low, double gain_high)
{
    const double wH = 2 * std::numbers::pi * wc / sr;
    const double g = gain_low / gain_high;
    const double g_sqrt = std::sqrt(g);

    double ah0 = std::tan(wH * 0.5) + g_sqrt;
    double ah1 = std::tan(wH * 0.5) - g_sqrt;
    double bh0 = (g * std::tan(wH * 0.5)) + g_sqrt;
    double bh1 = (g * std::tan(wH * 0.5)) - g_sqrt;

    std::array<double, 4> sos = {gain_high * bh0, gain_high * bh1, ah0, ah1};
    // Normalize the coefficients
    for (double& so : sos)
    {
        so /= ah0;
    }
    return sos;
}

Eigen::ArrayXcd Polyval(const Eigen::ArrayXd& p, const Eigen::ArrayXcd& x)
{
    Eigen::ArrayXcd result = Eigen::ArrayXcd::Zero(x.size());
    result += p[0];

    for (auto i = 1; i < p.size(); ++i)
    {
        result = x * result + p[i];
    }

    return result;
}

std::vector<double> freqz(std::span<const double> b, std::span<const double> a, std::span<const double> w, double sr)
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
        dig_w = Eigen::exp(std::complex(0.0, 1.0) * w_map * (-2.0 * std::numbers::pi_v<double> / sr));
    }
    else
    {
        dig_w = Eigen::exp(std::complex(0.0, 1.0) * w_map);
    }

    Eigen::Map<const Eigen::ArrayXd> b_map(b.data(), b.size());
    Eigen::ArrayXcd num = Polyval(b_map, dig_w);

    Eigen::Map<const Eigen::ArrayXd> a_map(a.data(), a.size());
    Eigen::ArrayXcd den = Polyval(a_map, dig_w);

    Eigen::ArrayXcd h_complex = num / den;

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
        beta = std::sqrt(std::abs((gb * gb) - 1.f) / std::abs((g * g) - (gb * gb))) * std::tan(b / 2);
    }

    const double b0 = (1 + g * beta) / (1 + beta);
    const double b1 = (-2 * std::cos(w0)) / (1 + beta);
    const double b2 = (1 - g * beta) / (1 + beta);

    const double a0 = 1.0;
    const double a1 = -2 * std::cos(w0) / (1 + beta);
    const double a2 = (1 - beta) / (1 + beta);

    return {b0, b1, b2, a0, a1, a2};
}

std::vector<double> aceq_d(std::span<const double> diff_mag, std::span<const double> freqs, double sr)
{
    constexpr uint32_t kNBands = 10;
    constexpr uint32_t kNumF = 19;
    constexpr double kGW = 0.3; // Gain factor at bandwidth

    if (diff_mag.size() != kNBands || freqs.size() != kNBands)
    {
        throw std::runtime_error("diff_mag and freqs must have size " + std::to_string(kNBands));
    }

    // array of center frequencies + intermediate frequencies
    std::array<double, kNumF> fc2 = {0};
    for (auto i = 0; i < freqs.size(); ++i)
    {
        fc2.at(i * 2) = freqs[i];
    }

    for (auto i = 1; i < fc2.size(); i += 2)
    {
        fc2.at(i) = std::sqrt(fc2.at(i - 1) * fc2.at(i + 1));
    }

    // Command gain frequencies in radians
    std::array<double, kNBands> wg = {0.0f};
    for (auto i = 0; i < wg.size(); ++i)
    {
        wg.at(i) = 2 * std::numbers::pi_v<double> * freqs[i] / sr;
    }

    // Center frequencies in radian for iterative design
    std::array<double, kNumF> wc = {0.0f};
    for (auto i = 0; i < fc2.size(); ++i)
    {
        wc.at(i) = 2 * std::numbers::pi_v<double> * fc2.at(i) / sr;
    }

    std::array<double, kNBands> bw = {0.0f};
    for (auto i = 0; i < wg.size(); ++i)
    {
        bw.at(i) = 1.5 * wg.at(i);
    }
    // Extra adjustment
    bw[7] *= 0.93;
    bw[8] *= 0.78;
    bw[9] = 0.76 * wg[9]; // todo: check if this is a bug in the original code

    Eigen::Array<double, kNBands, 1> G;
    G.setConstant(std::pow(10.0, kNumF / 20.0));

    Eigen::MatrixXd leak = InteractionMatrix(G, kGW, wg, wc, bw);

    Eigen::Map<const Eigen::Array<double, kNBands, 1>> Gdb(diff_mag.data(), diff_mag.size());

    Eigen::Vector<double, kNumF> Gdb2 = Eigen::VectorXd::Zero(kNumF);
    Gdb2(Eigen::seq(0, kNumF - 1, 2)) = Gdb;
    Gdb2(Eigen::seq(1, kNumF - 1, 2)) = (Gdb2(Eigen::seq(0, kNumF - 3, 2)) + Gdb2(Eigen::seq(2, kNumF - 1, 2))) / 2;

    // Solve least squares optmization problem
    Eigen::Vector<double, kNBands> solution = (leak * leak.transpose()).ldlt().solve(leak * Gdb2);

    Eigen::Array<double, kNBands, 1> goptdb = Eigen::pow(10.0, solution.array() / 20.0);
    Eigen::Array<double, kNBands, 1> gwopt = Eigen::pow(10.0, kGW * solution.array() / 20.0);

    constexpr int kNumIteration = 1;

    for (int iter = 0; iter < kNumIteration; ++iter)
    {
        Eigen::MatrixXd leak2 = InteractionMatrix(goptdb, kGW, wg, wc, bw);
        Eigen::Vector<double, kNBands> solution2 = (leak2 * leak2.transpose()).ldlt().solve(leak2 * Gdb2);

        solution2 /= 20.0;
        goptdb = Eigen::pow(10.0, solution2.array());
        gwopt = Eigen::pow(10.0, kGW * solution2.array());
    }

    std::vector<double> sos;
    for (auto i = 0; i < kNBands; ++i)
    {
        std::array<double, 6> coeffs = Pareq(goptdb[i], gwopt[i], wg.at(i), bw.at(i));
        sos.insert(sos.end(), coeffs.begin(), coeffs.end());
    }

    return sos;
}

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
    const auto kNBands = t60s.size();
    std::vector<double> freqs(kNBands, 0.0);
    constexpr double kUpperLimit = 16000.0f;
    for (auto i = 0; i < kNBands; ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0, static_cast<double>(kNBands - 1 - i));
    }

    std::vector<double> gains(kNBands, 0.0f);
    for (auto i = 0; i < kNBands; ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], delay / sr);
        gains[i] = 20.0 * std::log10(gains[i]);
    }

    std::vector<double> linear_gains(gains.size(), 0.0);
    for (auto i = 0; i < gains.size(); ++i)
    {
        linear_gains[i] = db2mag(gains[i]);
    }

    // Build first-order low shelf filter
    double gain_low = linear_gains[0];
    double gain_high = linear_gains[linear_gains.size() - 1];

    std::array<double, 4> shelf_sos = LowShelf(shelf_cutoff, sr, gain_low, gain_high);
    std::span shelf_sos_span{shelf_sos};

    auto b_coeffs = shelf_sos_span.first(2);
    auto a_coeffs = shelf_sos_span.last(2);
    std::vector<double> Hshelf = freqz(b_coeffs, a_coeffs, freqs, sr);

    std::vector<double> diff_mag(freqs.size(), 0.0f);
    for (auto i = 0; i < freqs.size(); ++i)
    {
        diff_mag[i] = gains[i] - 20 * std::log10(Hshelf[i]);
    }

    std::vector<double> sos_double;
    if (kNBands == 10) // octave bands
    {
        sos_double = aceq_d(diff_mag, freqs, sr);
    }

    assert(sos_double.size() == kNBands * 6);

    std::vector<double> sos(sos_double.size() + 6, 0.0f);

    // Copy the low shelf filter coefficients
    sos[0] = shelf_sos[0];
    sos[1] = shelf_sos[1];
    sos[2] = 0.0f;
    sos[3] = shelf_sos[2];
    sos[4] = shelf_sos[3];
    sos[5] = 0.0f;

    for (auto i = 0; i < sos_double.size(); ++i)
    {
        sos[i + 6] = sos_double[i];
    }

    return sos;
}

std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff)
{
    std::vector<double> t60s_d(t60s.begin(), t60s.end());
    std::vector<double> sos = GetTwoFilter_d(t60s_d, static_cast<double>(delay), static_cast<double>(sr), shelf_cutoff);

    std::vector<float> sos_f(sos.begin(), sos.end());

    return sos_f;
}

std::vector<float> DesignGraphicEQ(std::span<const float> mag, std::span<const float> freqs, float sr)
{
    std::vector<double> mag_d(mag.begin(), mag.end());
    std::vector<double> freqs_d(freqs.begin(), freqs.end());
    std::vector<double> sos = aceq_d(mag_d, freqs_d, static_cast<double>(sr));
    std::vector<float> sos_f(sos.begin(), sos.end());
    return sos_f;
}

} // namespace sfFDN
