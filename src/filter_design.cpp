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
#include <variant>
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
std::pair<float, float> DesignTwoBandAbsorption(const TwoBandFilterOptions& options)
{
    const float h_dc = Db2Mag(options.delay * RT602Slope(options.t60s[0], options.sample_rate));
    const float h_ny = Db2Mag(options.delay * RT602Slope(options.t60s[1], options.sample_rate));

    const float r = h_dc / h_ny;
    const float a = (1 - r) / (1 + r);
    const float b = (1 - a) * h_ny;
    return {b, a};
}

std::array<FilterCoefficients, 2> DesignThreeBandAbsorption(const ThreeBandFilterOptions& options)
{
    const float g_dc_db = options.delay * RT602Slope(options.t60s[0], options.sample_rate);
    const float g_mid_db = options.delay * RT602Slope(options.t60s[1], options.sample_rate);
    const float g_ny_db = options.delay * RT602Slope(options.t60s[2], options.sample_rate);

    auto low_shelf = sfFDN::LowShelfRBJ(options.freqs[0] / options.sample_rate, g_dc_db - g_mid_db, options.q);
    auto high_shelf = sfFDN::HighShelfRBJ(options.freqs[1] / options.sample_rate, g_ny_db - g_mid_db, options.q);

    const float g_mid_linear = Db2Mag(g_mid_db);
    // Apply mid gain to b coefficients of the low shelf filter
    low_shelf[0] *= g_mid_linear;
    low_shelf[1] *= g_mid_linear;
    low_shelf[2] *= g_mid_linear;

    std::array<FilterCoefficients, 2> sos = {{{.b0 = low_shelf[0],
                                               .b1 = low_shelf[1],
                                               .b2 = low_shelf[2],
                                               .a0 = low_shelf[3],
                                               .a1 = low_shelf[4],
                                               .a2 = low_shelf[5]},
                                              {.b0 = high_shelf[0],
                                               .b1 = high_shelf[1],
                                               .b2 = high_shelf[2],
                                               .a0 = high_shelf[3],
                                               .a1 = high_shelf[4],
                                               .a2 = high_shelf[5]}}};
    return sos;
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

std::array<FilterCoefficients, 11> DesignTenBandAbsorption(const TenBandFilterOptions& options)
{
    // The coefficients are computed in double precision, otherwise there is a significant loss of precision and the
    // filter is not as accurate as it could be.
    std::vector<double> gains(options.t60s.size(), 0.0f);
    for (auto i = 0u; i < gains.size(); ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / options.t60s[i]);
        gains[i] = std::pow(gains[i], options.delay / options.sample_rate);
        gains[i] = 20.0 * std::log10(gains[i]);
    }
    std::vector<double> freqs(options.t60s.size(), 0.0);
    constexpr double kUpperLimit = 16000.0f;
    for (auto i = 0u; i < options.t60s.size(); ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0, static_cast<double>(options.t60s.size() - 1 - i));
    }

    const std::vector<double> sos =
        GetTwoFilterImpl(gains, freqs, static_cast<double>(options.sample_rate), options.shelf_cutoff);

    std::array<FilterCoefficients, 11> sos_f{{}};
    assert(sos.size() == sos_f.size() * 6);
    sos_f.fill({});
    for (auto i = 0u; i < sos_f.size(); ++i)
    {
        sos_f[i].b0 = static_cast<float>(sos[6 * i]);
        sos_f[i].b1 = static_cast<float>(sos[6 * i + 1]);
        sos_f[i].b2 = static_cast<float>(sos[6 * i + 2]);
        sos_f[i].a0 = static_cast<float>(sos[6 * i + 3]);
        sos_f[i].a1 = static_cast<float>(sos[6 * i + 4]);
        sos_f[i].a2 = static_cast<float>(sos[6 * i + 5]);
    }

    return sos_f;
}

std::array<FilterCoefficients, 11> DesignGraphicEQ(const GraphicEQOptions& options)
{
    std::vector<double> gains(options.gains_db.begin(), options.gains_db.end());
    std::vector<double> freqs_d(options.freqs.begin(), options.freqs.end());

    const std::vector<double> sos = GetTwoFilterImpl(gains, freqs_d, static_cast<double>(options.sample_rate), 8000.0);

    std::array<FilterCoefficients, 11> sos_f{};
    assert(sos.size() == sos_f.size() * 6);
    sos_f.fill({});
    for (auto i = 0u; i < sos_f.size(); ++i)
    {
        sos_f[i].b0 = static_cast<float>(sos[6 * i]);
        sos_f[i].b1 = static_cast<float>(sos[6 * i + 1]);
        sos_f[i].b2 = static_cast<float>(sos[6 * i + 2]);
        sos_f[i].a0 = static_cast<float>(sos[6 * i + 3]);
        sos_f[i].a1 = static_cast<float>(sos[6 * i + 4]);
        sos_f[i].a2 = static_cast<float>(sos[6 * i + 5]);
    }
    return sos_f;
}

std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(attenuation_filter_variant_t options,
                                                            std::span<const float> delays)
{
    sfFDN::AttenuationFilterBankOptions fb_options;
    fb_options.filter_configs.resize(delays.size());
    for (size_t i = 0; i < delays.size(); ++i)
    {
        std::visit(sfFDN::overloaded{[&](auto& arg) { arg.delay = delays[i]; }}, options);
        fb_options.filter_configs[i] = options;
    }
    return CreateAttenuationFilterBank(fb_options);
}

std::unique_ptr<AudioProcessor> CreateAttenuationFilter(const attenuation_filter_variant_t& options)
{
    return std::visit(overloaded{[&](const HomogenousFilterOptions& config) -> std::unique_ptr<AudioProcessor> {
                                     float feedback_gain = Db2Mag(RT602Slope(config.t60, config.sample_rate));
                                     feedback_gain = std::pow(feedback_gain, config.delay);
                                     return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel,
                                                                                   std::vector<float>{feedback_gain});
                                 },
                                 [](const TwoBandFilterOptions& config) -> std::unique_ptr<AudioProcessor> {
                                     auto [b, a] = DesignTwoBandAbsorption(config);
                                     return std::make_unique<sfFDN::OnePoleFilter>(b, a);
                                 },
                                 [](const ThreeBandFilterOptions& config) -> std::unique_ptr<AudioProcessor> {
                                     auto sos = DesignThreeBandAbsorption(config);
                                     auto filter = std::make_unique<sfFDN::CascadedBiquads>();
                                     filter->SetCoefficients(sos);
                                     return filter;
                                 },
                                 [](const TenBandFilterOptions& config) -> std::unique_ptr<AudioProcessor> {
                                     auto sos = DesignTenBandAbsorption(config);
                                     auto filter = std::make_unique<sfFDN::CascadedBiquads>();
                                     filter->SetCoefficients(sos);
                                     return filter;
                                 }

                      },
                      options);
}

std::unique_ptr<AudioProcessor> CreateAttenuationFilterBank(const AttenuationFilterBankOptions& options)
{
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : options.filter_configs)
    {
        filter_bank->AddFilter(CreateAttenuationFilter(config));
    }
    return filter_bank;
}

} // namespace sfFDN