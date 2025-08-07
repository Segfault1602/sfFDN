#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_CASE("Accumulate")
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
    });
}

TEST_CASE("Add")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Add");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("Add", [&] {
        for (auto i = 0; i < 1000; ++i)
        {
            sfFDN::ArrayMath::Add(a, b, out);
            nanobench::doNotOptimizeAway(a);
        }
    });
}

TEST_CASE("Scale")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Scale");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("Scale", [&] {
        for (auto i = 0; i < 1000; ++i)
        {
            sfFDN::ArrayMath::Scale(a, 2.f, out);
            nanobench::doNotOptimizeAway(out);
        }
    });
}
TEST_CASE("ScaleAdd")
{
    constexpr uint32_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("ScaleAdd");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("ScaleAdd", [&] {
        for (auto i = 0; i < 1000; ++i)
        {
            sfFDN::ArrayMath::ScaleAdd(a, 2.f, b, out);
            nanobench::doNotOptimizeAway(out);
        }
    });
}