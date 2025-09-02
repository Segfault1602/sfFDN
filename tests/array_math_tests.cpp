#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <numeric>
#include <vector>

#include <array_math.h>

TEST_CASE("Accumulate")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 0.f);
    std::iota(a.begin(), a.end(), 0.f); // Fill with 0, 1, ..., N-1

    std::vector<float> b(N, 0.f);
    std::iota(b.begin(), b.end(), 1.f); // Fill with 1, 2, ..., N

    sfFDN::ArrayMath::Accumulate(a, b);
    for (auto i = 0; i < N; ++i)
    {
        REQUIRE_THAT(a[i], Catch::Matchers::WithinAbs(i + b[i], 0.0001));
    }
}

TEST_CASE("Add")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    sfFDN::ArrayMath::Add(a, b, out);
    for (auto i = 0; i < N; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(3.f, 0.0001));
    }
}

TEST_CASE("Scale")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    sfFDN::ArrayMath::Scale(a, 2.f, out);
    for (auto i = 0; i < N; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(2.f, 0.0001));
    }
}

TEST_CASE("ScaleAdd")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    sfFDN::ArrayMath::ScaleAdd(a, 2.f, b, out);
    for (auto i = 0; i < N; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(4.f, 0.0001));
    }
}

TEST_CASE("ScaleAccumulate")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    sfFDN::ArrayMath::ScaleAccumulate(a, 2.f, out);
    for (auto i = 0; i < N; ++i)
    {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(2.f, 0.0001));
    }
}
