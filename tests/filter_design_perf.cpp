#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <complex>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>

#include <Eigen/Core>

using namespace ankerl;
using namespace std::chrono_literals;

#include <filter_design_internal.h>
#include <sffdn/sffdn.h>

TEST_CASE("TwoFilter")
{
    constexpr float kSR = 48000;
    constexpr std::array<float, 10> kT60s = {2.5, 2.7, 2.5, 2.3, 2.3, 2.1, 1.7, 1.6, 1.2, 1.0};
    constexpr float kDelay = 1000;
    constexpr float kShelfCutoff = 8000.0f;

    nanobench::Bench bench;
    bench.title("GetTwoFilter");
    bench.minEpochIterations(10000);
    bench.timeUnit(1us, "us");

    bench.run("GetTwoFilter", [&] {
        auto coeffs = sfFDN::GetTwoFilter(kT60s, kDelay, kSR, kShelfCutoff);
        nanobench::doNotOptimizeAway(coeffs);
    });
}

namespace
{
template <size_t N>
Eigen::Array<std::complex<double>, N, 1> Polyval_(const Eigen::ArrayXd& p,
                                                  const Eigen::Array<std::complex<double>, N, 1>& x)
{
    Eigen::Array<std::complex<double>, N, 1> result = Eigen::Array<std::complex<double>, N, 1>::Zero();
    result += p[0];

    for (auto i = 1; i < p.size(); ++i)
    {
        result = x * result + p[i];
    }

    return result;
}

} // namespace

TEST_CASE("Polyval")
{
    constexpr size_t kN = 4069;
    constexpr double FS = 48000.0;

    std::vector<double> freqs(kN);
    for (size_t i = 0; i < kN; ++i)
    {
        freqs[i] = i * (FS / 2.0 / kN);
    }

    Eigen::Map<const Eigen::ArrayXd> w_map(freqs.data(), freqs.size());
    Eigen::ArrayXcd dig_w(w_map.size());
    dig_w = Eigen::exp(std::complex(0.0, 1.0) * w_map * (-2.0 * std::numbers::pi_v<double> / FS));

    Eigen::ArrayXd p(3);
    p << 0.5, -0.8, 0.2;

    nanobench::Bench bench;
    bench.title("Polyval");
    bench.minEpochIterations(10000);
    bench.relative(true);

    // bench.run("Eigen - dynamic", [&] {
    //     Eigen::ArrayXcd result = sfFDN::Polyval(p, dig_w);
    //     nanobench::doNotOptimizeAway(result);
    // });

    Eigen::Array<std::complex<double>, kN, 1> dig_w_arr;
    dig_w_arr = dig_w;

    bench.run("Eigen - template", [&] {
        Eigen::Array<std::complex<double>, kN, 1> result = Polyval_<kN>(p, dig_w_arr);
        nanobench::doNotOptimizeAway(result);
    });

    std::array<double, 3> p_arr = {0.5, -0.8, 0.2};
    std::vector<std::complex<double>> dig_w_std(kN);
    for (size_t i = 0; i < kN; ++i)
    {
        dig_w_std[i] = std::exp(std::complex<double>(0.0, 1.0) * freqs[i]);
    }

    bench.run("std - template", [&] {
        std::array<std::complex<double>, kN> result;
        sfFDN::Polyval<double>(p_arr, dig_w_std, result);
        nanobench::doNotOptimizeAway(result);
    });
}