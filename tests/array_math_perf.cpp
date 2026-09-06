#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

#include <array_math.h>

namespace
{
void RunArrayMathBenchmarks(uint32_t block_size, nanobench::Bench& bench)
{
    std::vector<float> a(block_size);
    std::vector<float> b(block_size);
    std::vector<float> out(block_size);

    for (auto i = 0u; i < block_size; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(block_size - i);
        out[i] = 0.f;
    }

    std::vector<float> accumulate = a;
    std::vector<float> add_inplace = a;
    std::vector<float> scale_inplace = a;
    std::vector<float> scale_accumulate = b;
    const std::string suffix = " N=" + std::to_string(block_size);

    bench.run("Accumulate (in-place)" + suffix, [&] {
        sfFDN::ArrayMath::Accumulate(accumulate, b);
        nanobench::doNotOptimizeAway(accumulate);
        nanobench::doNotOptimizeAway(b);
    });

    bench.run("Add" + suffix, [&] {
        sfFDN::ArrayMath::Add(a, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });

    bench.run("Add (in-place)" + suffix, [&] {
        sfFDN::ArrayMath::Add(add_inplace, b, add_inplace);
        nanobench::doNotOptimizeAway(add_inplace);
        nanobench::doNotOptimizeAway(b);
    });

    bench.run("Scale" + suffix, [&] {
        sfFDN::ArrayMath::Scale(a, 2.f, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(out);
    });

    bench.batch(2U * block_size);
    bench.run("Scale (in-place)" + suffix, [&] {
        sfFDN::ArrayMath::Scale(scale_inplace, 2.f, scale_inplace);
        sfFDN::ArrayMath::Scale(scale_inplace, 0.5f, scale_inplace);
        nanobench::doNotOptimizeAway(scale_inplace);
    });
    REQUIRE(std::ranges::all_of(scale_inplace, [](float value) { return std::isfinite(value); }));
    bench.batch(block_size);

    bench.run("ScaleAccumulate (in-place)" + suffix, [&] {
        sfFDN::ArrayMath::ScaleAccumulate(a, 2.f, scale_accumulate);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(scale_accumulate);
    });

    bench.run("Multiply" + suffix, [&] {
        sfFDN::ArrayMath::Multiply(a, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });

    bench.run("MultiplyAdd" + suffix, [&] {
        sfFDN::ArrayMath::MultiplyAdd(a, 2.f, b, out);
        nanobench::doNotOptimizeAway(a);
        nanobench::doNotOptimizeAway(b);
        nanobench::doNotOptimizeAway(out);
    });
}
} // namespace

TEST_CASE("ArrayMath", "[array_math]")
{
    nanobench::Bench bench;
    bench.title("ArrayMath perf");
    bench.timeUnit(1ns, "ns");
    bench.warmup(1000);
    bench.minEpochTime(20ms);

    for (const uint32_t block_size : std::array{32u, 64u, 128u, 256u})
    {
        bench.batch(block_size);
        RunArrayMathBenchmarks(block_size, bench);
    }
}