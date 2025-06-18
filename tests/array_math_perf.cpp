#include "doctest.h"
#include "nanobench.h"

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_SUITE_BEGIN("ArrayMath");

TEST_CASE("Accumulate")
{
    constexpr size_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);

    nanobench::Bench bench;
    bench.title("Accumulate");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("Accumulate", [&] {
        for (size_t i = 0; i < 1000; ++i)
        {
            fdn::ArrayMath::Accumulate(a, b);
            nanobench::doNotOptimizeAway(a);
        }
    });
}

TEST_CASE("Add")
{
    constexpr size_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Add");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("Add", [&] {
        for (size_t i = 0; i < 1000; ++i)
        {
            fdn::ArrayMath::Add(a, b, out);
            nanobench::doNotOptimizeAway(a);
        }
    });
}

TEST_CASE("Scale")
{
    constexpr size_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Scale");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("Scale", [&] {
        for (size_t i = 0; i < 1000; ++i)
        {
            fdn::ArrayMath::Scale(a, 2.f, out);
            nanobench::doNotOptimizeAway(out);
        }
    });
}
TEST_CASE("ScaleAdd")
{
    constexpr size_t N = 128;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("ScaleAdd");
    bench.minEpochIterations(1000);
    bench.batch(1000);

    bench.run("ScaleAdd", [&] {
        for (size_t i = 0; i < 1000; ++i)
        {
            fdn::ArrayMath::ScaleAdd(a, 2.f, b, out);
            nanobench::doNotOptimizeAway(out);
        }
    });
}

TEST_SUITE_END();