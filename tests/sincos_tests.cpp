#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "sincos.h"
#include "sine_table.h"

namespace
{

constexpr float kTau = 2.0f * std::numbers::pi_v<float>;
// Measured worst case for the table + Newton implementation is ~2.4e-7 (2 ulp at 1.0) on clang/arm64.
// The tolerance is set an order of magnitude above that so the test does not become flaky across
// compilers with different FP contraction (CI also builds MSVC), while still sitting ~9x below the
// ~9.4e-6 norm deficit that an unnormalized linearly-interpolated table would produce -- which is
// the defect this test exists to catch.
constexpr float kNormalizationTolerance = 1.0e-6f;
// The table-based implementation measures below 4.4e-7 absolute component error in this sweep.
constexpr float kAccuracyTolerance = 1.0e-6f;

} // namespace

TEST_CASE("SinCosUnit returns normalized pairs across every sine table interval")
{
    constexpr std::array<float, 4> kOffsets = {0.0f, 0.25f, 0.5f, 0.75f};

    for (size_t interval = 0; interval < sfFDN::kSineTableSize; ++interval)
    {
        for (const float offset : kOffsets)
        {
            const float radians = (static_cast<float>(interval) + offset) * (kTau / sfFDN::kSineTableSize);
            float sin_value = 0.0f;
            float cos_value = 0.0f;
            sfFDN::SinCosUnit(radians, sin_value, cos_value);

            const float norm_error = std::abs(((sin_value * sin_value) + (cos_value * cos_value)) - 1.0f);
            CHECK(norm_error < kNormalizationTolerance);
        }
    }
}

TEST_CASE("SinCosUnit remains accurate after normalization")
{
    constexpr std::array<float, 4> kOffsets = {0.0f, 0.25f, 0.5f, 0.75f};

    for (size_t interval = 0; interval < sfFDN::kSineTableSize; ++interval)
    {
        for (const float offset : kOffsets)
        {
            const float radians = (static_cast<float>(interval) + offset) * (kTau / sfFDN::kSineTableSize);
            float sin_value = 0.0f;
            float cos_value = 0.0f;
            sfFDN::SinCosUnit(radians, sin_value, cos_value);

            REQUIRE_THAT(sin_value, Catch::Matchers::WithinAbs(std::sin(radians), kAccuracyTolerance));
            REQUIRE_THAT(cos_value, Catch::Matchers::WithinAbs(std::cos(radians), kAccuracyTolerance));
        }
    }
}

TEST_CASE("SinCosUnit range-reduces finite angles")
{
    constexpr std::array<float, 6> kAngles = {
        -0.5f * std::numbers::pi_v<float>, 2.5f * std::numbers::pi_v<float>, -1000.0f, 1000.0f, -1.0e20f, 1.0e20f,
    };

    for (const float radians : kAngles)
    {
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sfFDN::SinCosUnit(radians, sin_value, cos_value);

        const float norm_error = std::abs(((sin_value * sin_value) + (cos_value * cos_value)) - 1.0f);
        REQUIRE(norm_error < kNormalizationTolerance);
    }
}

TEST_CASE("SinCosUnit is component-accurate across its contracted input range")
{
    constexpr std::array<float, 6> kAngles = {
        -2.0f * kTau, -1.5f * kTau, -0.5f * kTau, 0.5f * kTau, 1.5f * kTau, 2.0f * kTau,
    };

    for (const float radians : kAngles)
    {
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sfFDN::SinCosUnit(radians, sin_value, cos_value);

        REQUIRE_THAT(sin_value, Catch::Matchers::WithinAbs(std::sin(radians), kAccuracyTolerance));
        REQUIRE_THAT(cos_value, Catch::Matchers::WithinAbs(std::cos(radians), kAccuracyTolerance));
    }
}

TEST_CASE("SinCosUnit selects signs in all quadrants")
{
    constexpr std::array<float, 4> kAngles = {
        0.25f * std::numbers::pi_v<float>,
        0.75f * std::numbers::pi_v<float>,
        1.25f * std::numbers::pi_v<float>,
        1.75f * std::numbers::pi_v<float>,
    };

    constexpr std::array<float, 4> kExpectedSineSigns = {1.0f, 1.0f, -1.0f, -1.0f};
    constexpr std::array<float, 4> kExpectedCosineSigns = {1.0f, -1.0f, -1.0f, 1.0f};

    for (size_t quadrant = 0; quadrant < kAngles.size(); ++quadrant)
    {
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sfFDN::SinCosUnit(kAngles[quadrant], sin_value, cos_value);

        REQUIRE(sin_value * kExpectedSineSigns[quadrant] > 0.0f);
        REQUIRE(cos_value * kExpectedCosineSigns[quadrant] > 0.0f);
    }
}
