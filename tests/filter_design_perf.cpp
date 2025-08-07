#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

using namespace ankerl;
using namespace std::chrono_literals;

#include <sffdn/sffdn.h>

TEST_CASE("TwoFilter")
{
    constexpr float kSR = 48000;
    constexpr std::array<float, 10> kT60s = {2.5, 2.7, 2.5, 2.3, 2.3, 2.1, 1.7, 1.6, 1.2, 1.0};
    constexpr float kDelay = 1000;
    constexpr float kShelfCutoff = 8000.0f;

    nanobench::Bench bench;
    bench.title("GetTwoFilter");
    bench.minEpochIterations(1000);
    bench.timeUnit(1us, "us");

    bench.run("GetTwoFilter", [&] {
        auto coeffs = sfFDN::GetTwoFilter(kT60s, kDelay, kSR, kShelfCutoff);
        nanobench::doNotOptimizeAway(coeffs);
    });
}