#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_CASE("Accumulate", "[ArrayMath]")
{
    constexpr uint32_t kSize = 128;
    std::vector<float> a(kSize, 1.f);
    std::vector<float> b(kSize, 2.f);
    std::vector<float> out(kSize, 0.f);

    nanobench::Bench bench;
    bench.title("Accumulate");
    bench.minEpochIterations(5000000);

    bench.run("Accumulate", [&] {
        sfFDN::ArrayMath::Accumulate(a, b);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
    });

    bench.run("Add", [&] {
        sfFDN::ArrayMath::Add(a, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });

    bench.run("Scale", [&] {
        sfFDN::ArrayMath::Scale(a, 2.f, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(out);
    });

    bench.run("ScaleAccumulate", [&] {
        sfFDN::ArrayMath::ScaleAccumulate(a, 2.f, b);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
    });
}