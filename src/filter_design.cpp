#include "sffdn/filter_design.h"

#include "filter_design_internal.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter.h"
#include "sffdn/filterbank.h"
#include "sffdn/parallel_gains.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
template <typename T>
T Db2Mag(T x)
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
    for (auto i = 0u; i < x.size(); ++i)
    {
        out[i] = 20.0 * std::log10(x[i]);
    }
}

template <typename T>
void Freqz(std::span<const T> b, std::span<const T> a, std::span<std::complex<T>> w, std::span<T> result)
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

template <size_t kNBands, size_t kNFreqs>
Eigen::MatrixXd InteractionMatrix(std::span<const double> gains, double gain_factor,
                                  std::span<const double> command_frequencies,
                                  std::span<const double> design_frequencies, std::span<const double> bandwidths)
{
    Eigen::MatrixXd leak = Eigen::MatrixXd::Zero(kNBands, kNFreqs);

    std::array<double, kNBands> gains_db{};
    ToDb<double>(gains, gains_db);

    const double gdb_abs_sum = std::accumulate(gains_db.begin(), gains_db.end(), 0.0,
                                               [](double sum, double val) { return sum + std::abs(val); });
    if (gdb_abs_sum <= 1e-10)
    {
        for (int i = 0; i < kNBands; ++i)
        {
            leak(i, i * 2) = 1;
        }
        return leak;
    }

    std::array<double, kNBands> gains_linear{};
    std::ranges::transform(gains_db, gains_linear.begin(),
                           [gain_factor](double val) -> double { return Db2Mag(gain_factor * val); });

    std::array<std::complex<double>, kNFreqs> dig_w_arr{};
    for (auto [w, f] : std::views::zip(dig_w_arr, design_frequencies))
    {
        w = std::exp(std::complex<double>(0.0, 1.0) * f);
    }

    for (auto i = 0u; i < kNBands; ++i)
    {
        std::array<double, 6> sos = sfFDN::Pareq(gains[i], gains_linear[i], command_frequencies[i], bandwidths[i]);
        auto sos_span = std::span<double>(sos);
        auto num = sos_span.first(3);
        auto den = sos_span.last(3);
        std::array<double, kNFreqs> filter_response{};
        Freqz<double>(num, den, dig_w_arr, filter_response);

        for (auto j = 0u; j < filter_response.size(); ++j)
        {
            leak(i, j) = (20.0 * std::log10(filter_response[j])) / gains_db[i];
        }
    }

    return leak;
}

template <size_t kNBands>
std::vector<double> Aceq(std::span<const double> diff_mag, std::span<const double> freqs, double sr)
{
    if (diff_mag.size() != kNBands || freqs.size() != kNBands)
    {
        throw std::runtime_error("diff_mag and freqs must have size " + std::to_string(kNBands));
    }

    constexpr size_t kNFreqs = (kNBands * 2) - 1;
    constexpr double kGW = 0.3; // Gain factor at bandwidth

    // array of center frequencies + intermediate frequencies
    std::array<double, kNFreqs> fc2 = {0};
    for (auto i = 0u; i < freqs.size(); ++i)
    {
        fc2.at(i * 2) = freqs[i];
    }

    for (auto i = 1; i < fc2.size(); i += 2)
    {
        fc2.at(i) = std::sqrt(fc2.at(i - 1) * fc2.at(i + 1));
    }

    // Command gain frequencies in radians
    std::array<double, kNBands> wg = {0.0f};
    for (auto [w, f] : std::views::zip(wg, freqs))
    {
        w = 2 * std::numbers::pi_v<double> * f / sr;
    }

    // Center frequencies in radian for iterative design
    std::array<double, kNFreqs> wc = {0.0f};
    for (auto [w, f] : std::views::zip(wc, fc2))
    {
        w = 2 * std::numbers::pi_v<double> * f / sr;
    }

    std::array<double, kNBands> bw = {0.0f};
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

    std::array<double, kNBands> gains_db{};
    gains_db.fill(std::pow(10.0, kNFreqs / 20.0));

    auto leak = InteractionMatrix<kNBands, kNFreqs>(gains_db, kGW, wg, wc, bw);

    const Eigen::Map<const Eigen::ArrayXd> diff_mag_map(diff_mag.data(), diff_mag.size());

    Eigen::VectorXd gains_db_2 = Eigen::VectorXd::Zero(kNFreqs);
    gains_db_2(Eigen::seq(0, kNFreqs - 1, 2)) = diff_mag_map;
    gains_db_2(Eigen::seq(1, kNFreqs - 1, 2)) =
        (gains_db_2(Eigen::seq(0, kNFreqs - 3, 2)) + gains_db_2(Eigen::seq(2, kNFreqs - 1, 2))) / 2;

    // Solve least squares optmization problem
    Eigen::VectorXd solution = (leak * leak.transpose()).ldlt().solve(leak * gains_db_2);

    std::array<double, kNBands> goptdb{};
    Eigen::Map<Eigen::ArrayXd> goptdb_map(goptdb.data(), goptdb.size());
    goptdb_map = Eigen::pow(10.0, solution.array() / 20);

    Eigen::ArrayXd gwopt = Eigen::pow(10.0, kGW * solution.array() / 20.0);

    Eigen::MatrixXd leak2 = InteractionMatrix<kNBands, kNFreqs>(goptdb, kGW, wg, wc, bw);
    Eigen::VectorXd solution2 = (leak2 * leak2.transpose()).ldlt().solve(leak2 * gains_db_2);

    goptdb_map = Eigen::pow(10.0, solution2.array() / 20);
    gwopt = Eigen::pow(10.0, kGW * solution2.array() / 20);

    std::vector<double> sos;
    for (auto i = 0u; i < kNBands; ++i)
    {
        std::array<double, 6> coeffs = sfFDN::Pareq(goptdb[i], gwopt[i], wg.at(i), bw.at(i));
        sos.insert(sos.end(), coeffs.begin(), coeffs.end());
    }

    return sos;
}

std::vector<double> GetTwoFilterImpl(std::span<const double> gains, std::span<const double> freqs, double sr,
                                     double shelf_cutoff)
{
    constexpr size_t kNBands = 10;

    if (gains.size() != kNBands)
    {
        throw std::runtime_error("gains must have size " + std::to_string(kNBands));
    }

    std::vector<double> linear_gains(gains.size(), 0.0);
    for (auto i = 0u; i < gains.size(); ++i)
    {
        linear_gains[i] = Db2Mag(gains[i]);
    }

    // Build first-order low shelf filter
    const double gain_low = linear_gains[0];
    const double gain_high = linear_gains[linear_gains.size() - 1];

    std::array<double, 4> shelf_sos = sfFDN::LowShelf(shelf_cutoff, sr, gain_low, gain_high);
    const std::span shelf_sos_span{shelf_sos};

    std::array<double, 3> b_coeffs = {shelf_sos[0] / shelf_sos[2], shelf_sos[1] / shelf_sos[2], 0.0f};
    std::array<double, 3> a_coeffs = {1.0f, shelf_sos[3] / shelf_sos[2], 0.0f};

    std::vector<std::complex<double>> dig_w(kNBands);
    for (size_t i = 0; i < kNBands; ++i)
    {
        dig_w[i] = std::exp(std::complex<double>(0.0, 1.0) * freqs[i] * (-2 * std::numbers::pi_v<double> / sr));
    }

    std::array<double, kNBands> h_shelf{};
    Freqz<double>(b_coeffs, a_coeffs, dig_w, h_shelf);

    std::vector<double> diff_mag(freqs.size(), 0.0f);
    for (auto i = 0u; i < freqs.size(); ++i)
    {
        diff_mag[i] = gains[i] - 20 * std::log10(h_shelf[i]);
    }

    std::vector<double> sos_t;
    if (kNBands == 10) // octave bands
    {
        sos_t = Aceq<kNBands>(diff_mag, freqs, sr);
    }

    assert(sos_t.size() == kNBands * 6);

    std::vector<double> sos(sos_t.size() + 6, 0.0f);

    // Copy the low shelf filter coefficients
    sos[0] = shelf_sos[0] / shelf_sos[2];
    sos[1] = shelf_sos[1] / shelf_sos[2];
    sos[2] = 0.0f;
    sos[3] = 1.f;
    sos[4] = shelf_sos[3] / shelf_sos[2];
    sos[5] = 0.0f;

    for (auto i = 0u; i < sos_t.size(); ++i)
    {
        sos[i + 6] = sos_t[i];
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
    const float h_dc = Db2Mag(delay * RT602Slope(t60_dc, sr));
    const float h_ny = Db2Mag(delay * RT602Slope(t60_ny, sr));

    const float r = h_dc / h_ny;
    a = (1 - r) / (1 + r);
    b = (1 - a) * h_ny;
}

std::vector<double> GetTwoFilter_d(std::span<const double> t60s, double delay, double sr, double shelf_cutoff)
{
    std::vector<double> gains(t60s.size(), 0.0f);
    for (auto i = 0u; i < gains.size(); ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], delay / sr);
        gains[i] = 20.0 * std::log10(gains[i]);
    }

    std::vector<double> freqs(t60s.size(), 0.0);
    constexpr double kUpperLimit = 16000.0f;
    for (auto i = 0u; i < t60s.size(); ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0, static_cast<double>(t60s.size() - 1 - i));
    }

    return GetTwoFilterImpl(gains, freqs, sr, shelf_cutoff);
}

std::vector<float> GetTwoFilter(std::span<const float> t60s, float delay, float sr, float shelf_cutoff)
{
    // The coefficients are computed in double precision, otherwise there is a significant loss of precision and the
    // filter is not as accurate as it could be.
    std::vector<double> gains(t60s.size(), 0.0f);
    for (auto i = 0u; i < gains.size(); ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], delay / sr);
        gains[i] = 20.0 * std::log10(gains[i]);
    }
    std::vector<double> freqs(t60s.size(), 0.0);
    constexpr double kUpperLimit = 16000.0f;
    for (auto i = 0u; i < t60s.size(); ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0, static_cast<double>(t60s.size() - 1 - i));
    }

    const std::vector<double> sos = GetTwoFilterImpl(gains, freqs, static_cast<double>(sr), shelf_cutoff);

    std::vector<float> sos_f;
    sos_f.reserve(sos.size());
    for (auto s : sos)
    {
        sos_f.push_back(static_cast<float>(s));
    }

    return sos_f;
}

std::vector<float> DesignGraphicEQ(std::span<const float> mag, std::span<const float> freqs, float sr)
{
    if (mag.size() != 10 || freqs.size() != 10)
    {
        throw std::runtime_error("mag and freqs must have size 10");
    }

    std::vector<double> gains(mag.begin(), mag.end());
    std::vector<double> freqs_d(freqs.begin(), freqs.end());

    const std::vector<double> sos = GetTwoFilterImpl(gains, freqs_d, static_cast<double>(sr), 8000.0);

    std::vector<float> sos_f;
    sos_f.reserve(sos.size());
    for (auto s : sos)
    {
        sos_f.push_back(static_cast<float>(s));
    }
    return sos_f;
}

std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(std::span<const float> t60s,
                                                            std::span<const uint32_t> delays, float sample_rate)
{

    if (t60s.size() == 1) // Proportional attenuation
    {
        const auto feedback_gain = Db2Mag(RT602Slope(t60s[0], sample_rate));
        std::vector<float> proportional_fb_gains(delays.size(), 0.f);
        for (size_t i = 0; i < delays.size(); ++i)
        {
            proportional_fb_gains[i] = std::powf(feedback_gain, static_cast<float>(delays[i]));
        }

        return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel, proportional_fb_gains);
    }

    if (t60s.size() == 2) // One-pole absorption filter
    {
        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (auto delay : delays)
        {
            auto onepole_filter = std::make_unique<sfFDN::OnePoleFilter>();
            onepole_filter->SetT60s(t60s[0], t60s[1], delay, sample_rate);
            filter_bank->AddFilter(std::move(onepole_filter));
        }

        return filter_bank;
    }

    if (t60s.size() == 10) // Two-filter attenuation
    {
        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (auto delay : delays)
        {
            std::vector<float> sos = GetTwoFilter(t60s, static_cast<float>(delay), sample_rate);
            const size_t num_stages = sos.size() / 6;
            auto filter = std::make_unique<sfFDN::CascadedBiquads>();
            filter->SetCoefficients(num_stages, sos);
            filter_bank->AddFilter(std::move(filter));
        }

        return filter_bank;
    }

    return nullptr;
}
} // namespace sfFDN