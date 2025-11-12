#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

TEST_CASE("ArrayMath")
{
    constexpr uint32_t kSize = 128;
    alignas(32) std::array<float, kSize> a{};
    alignas(32) std::array<float, kSize> b{};
    alignas(32) std::array<float, kSize> out{};

    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(kSize - i);
        out[i] = 0.f;
    }

    nanobench::Bench bench;
    bench.title("ArrayMath perf");
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

    bench.run("MultiplyAdd", [&] {
        sfFDN::ArrayMath::MultiplyAdd(a, 2.f, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });
}