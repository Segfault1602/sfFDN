#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "sincos.h"

#include <array>
#include <chrono>
#include <numbers>

using namespace ankerl;

TEST_CASE("SinCosUnitPerf", "[sincos]")
{
    constexpr std::array kAngles = {
        -std::numbers::pi_v<float>,         -0.7F * std::numbers::pi_v<float>,
        -0.25F * std::numbers::pi_v<float>, 0.0F,
        0.25F * std::numbers::pi_v<float>,  0.7F * std::numbers::pi_v<float>,
        std::numbers::pi_v<float>,          1.75F * std::numbers::pi_v<float>,
    };

    float sine = 0.0F;
    float cosine = 0.0F;
    nanobench::Bench bench;
    bench.title("SinCosUnit");
    bench.timeUnit(std::chrono::nanoseconds(1), "ns");
    bench.batch(kAngles.size());
    bench.minEpochIterations(1'000'000);
    bench.run("representative angles", [&] {
        float output_sum = 0.0F;
        for (const float angle : kAngles)
        {
            sfFDN::SinCosUnit(angle, sine, cosine);
            output_sum += sine + cosine;
        }
        nanobench::doNotOptimizeAway(output_sum);
    });
}
