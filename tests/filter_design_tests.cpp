#include <array>
#include <print>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sffdn/sffdn.h>

#include "filter_design_internal.h"

TEST_CASE("TwoFilter")
{
    constexpr float kSR = 48000;
    constexpr std::array<double, 10> kT60s = {2.5, 2.7, 2.5, 2.3, 2.3, 2.1, 1.7, 1.6, 1.2, 1.0};
    constexpr float kDelay = 1000;
    constexpr float kShelfCutoff = 8000.0f;

    auto coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    constexpr std::array<double, 66> expected_sos = {0.893771859789158, -0.221475677049208, 0,
                                                     1.000000000000000, -0.287867764010237, 0,
                                                     0.999997246298566, -1.995971686001405, 0.995991139250743,
                                                     1.000000000000000, -1.995971686001405, 0.995988385549309,
                                                     1.000019428792040, -1.991950901493539, 0.991998137725919,
                                                     1.000000000000000, -1.991950901493539, 0.992017566517959,
                                                     1.000000502846831, -1.983794551519546, 0.984059639110872,
                                                     1.000000000000000, -1.983794551519546, 0.984060141957704,
                                                     0.999934366930738, -1.967248185325194, 0.968367671956371,
                                                     1.000000000000000, -1.967248185325194, 0.968302038887109,
                                                     0.999949209689687, -1.933484223418133, 0.937683634383009,
                                                     1.000000000000000, -1.933484223418133, 0.937632844072696,
                                                     0.999814817024837, -1.862609157253365, 0.878866721222699,
                                                     1.000000000000000, -1.862609157253365, 0.878681538247535,
                                                     0.997843030899871, -1.707422645052083, 0.769810963415330,
                                                     1.000000000000000, -1.707422645052083, 0.767653994315201,
                                                     0.998591003286513, -1.384314044906827, 0.599877169653359,
                                                     1.000000000000000, -1.384314044906827, 0.598468172939872,
                                                     0.995185270835553, -0.683231993084642, 0.371278715333731,
                                                     1.000000000000000, -0.683231993084642, 0.366463986169284,
                                                     0.997388335671291, 0.598553448808867,  0.199718561946444,
                                                     1.000000000000000, 0.598553448808867,  0.197106897617735};

    for (auto i = 0; i < coeffs.size(); ++i)
    {
        REQUIRE(coeffs[i] == Catch::Approx(expected_sos.at(i)).epsilon(1e-7));
    }
}

TEST_CASE("Polyval")
{
    Eigen::ArrayXd freqs(10);
    freqs << 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000;
    Eigen::ArrayXcd dig_w(freqs.size());

    dig_w = Eigen::exp(std::complex(0.0, 1.0) * freqs);

    Eigen::ArrayXd p(3);
    p << 0.5, -0.8, 0.2;

    Eigen::ArrayXcd result = sfFDN::Polyval(p, dig_w);

    Eigen::ArrayXcd expected(10);
    expected << std::complex(-0.116292474735830, -0.030764807808418),
        std::complex(-0.162494939592148, -0.047383785262679), std::complex(-0.309677457072758, +0.007568357580022),
        std::complex(-0.434715280943946, +0.542536512972206), std::complex(1.188268956890534, +0.787657214523982),
        std::complex(-0.433633035582978, -0.196483880217534), std::complex(0.128994159506051, -1.085783500471624),
        std::complex(0.816780131394544, +1.045724551283134), std::complex(-0.348206819242412, -0.732770892795190),
        std::complex(1.475942296183083, -0.234683626155909);

    for (int i = 0; i < result.size(); ++i)
    {
        REQUIRE(result[i].imag() == Catch::Approx(expected[i].imag()).epsilon(1e-7));
        REQUIRE(result[i].real() == Catch::Approx(expected[i].real()).epsilon(1e-7));
    }
}
