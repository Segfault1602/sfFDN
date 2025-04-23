#include "doctest.h"
#include "nanobench.h"

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_SUITE_BEGIN("ArrayMath");

TEST_CASE("Accumulate")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);

    nanobench::Bench bench;
    bench.title("Accumulate");
    bench.minEpochIterations(100);
    bench.timeUnit(1us, "us");

    bench.run("Accumulate", [&] { fdn::ArrayMath::Accumulate(a, b); });
}

TEST_CASE("Add")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Add");
    bench.minEpochIterations(100);
    bench.timeUnit(1us, "us");

    bench.run("Add", [&] { fdn::ArrayMath::Add(a, b, out); });
}

TEST_CASE("Scale")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("Scale");
    bench.minEpochIterations(100);
    bench.timeUnit(1us, "us");

    bench.run("Scale", [&] { fdn::ArrayMath::Scale(a, 2.f, out); });
}
TEST_CASE("ScaleAdd")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    nanobench::Bench bench;
    bench.title("ScaleAdd");
    bench.minEpochIterations(100);
    bench.timeUnit(1us, "us");

    bench.run("ScaleAdd", [&] { fdn::ArrayMath::ScaleAdd(a, 2.f, b, out); });
}

TEST_SUITE_END();