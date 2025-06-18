#include <doctest.h>

#include <iostream>
#include <vector>

#include <array_math.h>

using namespace fdn;

TEST_SUITE_BEGIN("ArrayMath");

TEST_CASE("Accumulate")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);

    ArrayMath::Accumulate(a, b);
    for (size_t i = 0; i < N; ++i)
    {
        CHECK(a[i] == doctest::Approx(3.f));
    }
}

TEST_CASE("Add")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::Add(a, b, out);
    for (size_t i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(3.f));
    }
}

TEST_CASE("Scale")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::Scale(a, 2.f, out);
    for (size_t i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(2.f));
    }
}

TEST_CASE("ScaleAdd")
{
    constexpr size_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::ScaleAdd(a, 2.f, b, out);
    for (size_t i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(4.f));
    }
}

TEST_SUITE_END();