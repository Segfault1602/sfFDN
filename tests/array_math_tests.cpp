#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include <array_math.h>

TEST_CASE("Accumulate")
{
    constexpr uint32_t kSize = 1024;
    std::array<float, kSize> a{};

    std::array<float, kSize> b{};

    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = i;
        b[i] = i + 1;
    }

    sfFDN::ArrayMath::Accumulate(a, b);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(a[i], Catch::Matchers::WithinAbs(i + b[i], 0.0001));
    }
}

TEST_CASE("Add")
{
    constexpr uint32_t kSize = 1024;
    std::vector<float> a(kSize);
    std::vector<float> b(kSize);
    std::vector<float> out(kSize, 0.f);
    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = 3.f - (static_cast<float>(i) * 0.125f);
    }

    sfFDN::ArrayMath::Add(a, b, out);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(a[i] + b[i], 0.0001));
    }
}

TEST_CASE("Scale")
{
    constexpr uint32_t kSize = 1024;
    std::vector<float> a(kSize);
    std::vector<float> out(kSize, 0.f);
    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i) / 32.f;
    }

    sfFDN::ArrayMath::Scale(a, 2.f, out);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(a[i] * 2.f, 0.0001));
    }
}

TEST_CASE("ScaleAccumulate")
{
    constexpr uint32_t kSize = 1024;
    std::vector<float> a(kSize);
    std::vector<float> out(kSize);
    std::vector<float> expected(kSize);
    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i) * 0.25f;
        out[i] = 5.f - (static_cast<float>(i) * 0.125f);
        expected[i] = out[i] + (a[i] * 2.f);
    }

    sfFDN::ArrayMath::ScaleAccumulate(a, 2.f, out);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(expected[i], 0.0001));
    }
}

TEST_CASE("Multiply")
{
    constexpr uint32_t kSize = 1024;
    std::vector<float> a(kSize);
    std::vector<float> b(kSize);
    std::vector<float> out(kSize, 0.f);
    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = 2.f - (static_cast<float>(i) * 0.125f);
    }

    sfFDN::ArrayMath::Multiply(a, b, out);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(a[i] * b[i], 0.0001));
    }
}

TEST_CASE("MultiplyAdd")
{
    constexpr uint32_t kSize = 1024;
    std::vector<float> a(kSize);
    std::vector<float> c(kSize);
    std::vector<float> out(kSize, 0.f);
    for (auto i = 0u; i < kSize; ++i)
    {
        a[i] = static_cast<float>(i) * 0.25f;
        c[i] = 3.f - (static_cast<float>(i) * 0.125f);
    }

    sfFDN::ArrayMath::MultiplyAdd(a, 4.f, c, out);
    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs((a[i] * 4.f) + c[i], 0.0001));
    }
}
