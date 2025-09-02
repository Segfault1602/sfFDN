#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_CASE("Accumulate", "[ArrayMath]")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);

    nanobench::Bench bench;
    bench.title("Accumulate");
    bench.minEpochIterations(5000000);

    bench.run("Accumulate", [&] {
        sfFDN::ArrayMath::Accumulate(a, b);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
    });
}

TEST_CASE("Add", "[ArrayMath]")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Add");
    bench.minEpochIterations(5000000);

    bench.run("Add", [&] {
        sfFDN::ArrayMath::Add(a, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });
}

TEST_CASE("Scale", "[ArrayMath]")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Scale");
    bench.minEpochIterations(5000000);

    bench.run("Scale", [&] {
        sfFDN::ArrayMath::Scale(a, 2.f, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(out);
    });
}

TEST_CASE("ScaleAdd", "[ArrayMath]")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("ScaleAdd");
    bench.minEpochIterations(5000000);

    bench.run("ScaleAdd", [&] {
        sfFDN::ArrayMath::ScaleAdd(a, 2.f, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });
}

TEST_CASE("ScaleAccumulate", "[ArrayMath]")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);

    nanobench::Bench bench;
    bench.title("ScaleAccumulate");
    bench.minEpochIterations(5000000);

    bench.run("ScaleAccumulate", [&] {
        sfFDN::ArrayMath::ScaleAccumulate(a, 2.f, b);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
    });
}