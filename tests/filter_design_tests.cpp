#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <print>
#include <ranges>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sffdn/sffdn.h>

#include "filter_design_internal.h"

TEST_CASE("TwoFilter")
{
    constexpr float kSR = 48000;
    constexpr std::array<double, 10> kT60s = {2.5, 2.7, 2.5, 2.3, 2.3, 2.1, 1.7, 1.6, 1.2, 1.0};
    // constexpr std::array<double, 10> kT60s = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    constexpr float kDelay = 1619;
    constexpr float kShelfCutoff = 8000.0f;

    std::vector<double> coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    // clang-format off
    constexpr std::array<double, 66> expected_sos = {
        0.833753922053558, -0.196112500418774, 0, 1.000000000000000, -0.300074975165566, 0,
        0.999995541761545, -1.995969983773138, 0.995991141545255, 1.000000000000000, -1.995969983773138, 0.995986683306800,
        1.000031455441331, -1.991962851696166, 0.991998061679195, 1.000000000000000, -1.991962851696166, 0.992029517120525,
        1.000000814161373, -1.983794860302083, 0.984059636620207, 1.000000000000000, -1.983794860302083, 0.984060450781580,
        0.999893742170679, -1.967208144896469, 0.968368234838074, 1.000000000000000, -1.967208144896469, 0.968261977008753,
        0.999917777296403, -1.933453811844555, 0.937684589949490, 1.000000000000000, -1.933453811844555, 0.937602367245893,
        0.999700260098036, -1.862502303348346, 0.878873502206346, 1.000000000000000, -1.862502303348346, 0.878573762304382,
        0.996510690312741, -1.706273726254403, 0.769953855737997, 1.000000000000000, -1.706273726254403, 0.766464546050738,
        0.997725098051978, -1.383712253277730, 0.600048185769784, 1.000000000000000, -1.383712253277730, 0.597773283821762,
        0.992215955533026, -0.682207683428299, 0.372199411323571, 1.000000000000000, -0.682207683428299, 0.364415366856597,
        0.995766905324853, 0.598066031393632, 0.200365157462411, 1.000000000000000, 0.598066031393632, 0.196132062787264
    };
    // clang-format on

    for (auto i = 0; i < coeffs.size(); ++i)
    {
        REQUIRE_THAT(coeffs[i], Catch::Matchers::WithinAbs(expected_sos.at(i), 1e-13));
    }

    std::array<float, 10> t60s_f;
    for (size_t i = 0; i < kT60s.size(); ++i)
    {
        t60s_f[i] = static_cast<float>(kT60s[i]);
    }

    auto float_coeffs = sfFDN::GetTwoFilter(t60s_f, kDelay, kSR, kShelfCutoff);
    for (auto i = 0; i < coeffs.size(); ++i)
    {
        REQUIRE_THAT(float_coeffs[i], Catch::Matchers::WithinAbs(expected_sos.at(i), 1e-7));
    }
}

TEST_CASE("Polyval")
{
    constexpr size_t kN = 10;
    std::array<double, kN> freqs = {31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
    std::array<std::complex<double>, kN> dig_w;

    for (auto [w, f] : std::views::zip(dig_w, freqs))
    {
        w = std::exp(std::complex<double>(0.0, 1.0) * f);
    }

    std::array p = {0.5, -0.8, 0.2};

    std::array<std::complex<double>, kN> result;
    sfFDN::Polyval<double>(p, dig_w, result);

    std::array<std::complex<double>, kN> expected = {std::complex<double>(-0.116292474735830, -0.030764807808418),
                                                     std::complex<double>(-0.162494939592148, -0.047383785262679),
                                                     std::complex<double>(-0.309677457072758, +0.007568357580022),
                                                     std::complex<double>(-0.434715280943946, +0.542536512972206),
                                                     std::complex<double>(1.188268956890534, +0.787657214523982),
                                                     std::complex<double>(-0.433633035582978, -0.196483880217534),
                                                     std::complex<double>(0.128994159506051, -1.085783500471624),
                                                     std::complex<double>(0.816780131394544, +1.045724551283134),
                                                     std::complex<double>(-0.348206819242412, -0.732770892795190),
                                                     std::complex<double>(1.475942296183083, -0.234683626155909)};

    for (auto [res, exp] : std::views::zip(result, expected))
    {
        REQUIRE_THAT(res.imag(), Catch::Matchers::WithinAbs(exp.imag(), 1e-14));
        REQUIRE_THAT(res.real(), Catch::Matchers::WithinAbs(exp.real(), 1e-14));
    }
}
