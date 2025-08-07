#include <doctest.h>

#include <numeric>
#include <vector>

#include <array_math.h>

using namespace sfFDN;

TEST_SUITE_BEGIN("ArrayMath");

TEST_CASE("Accumulate")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 0.f);
    std::iota(a.begin(), a.end(), 0.f); // Fill with 0, 1, ..., N-1

    std::vector<float> b(N, 0.f);
    std::iota(b.begin(), b.end(), 1.f); // Fill with 1, 2, ..., N

    ArrayMath::Accumulate(a, b);
    for (auto i = 0; i < N; ++i)
    {
        CHECK(a[i] == doctest::Approx(i + b[i]));
    }
}

TEST_CASE("Add")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::Add(a, b, out);
    for (auto i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(3.f));
    }
}

TEST_CASE("Scale")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::Scale(a, 2.f, out);
    for (auto i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(2.f));
    }
}

TEST_CASE("ScaleAdd")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> b(N, 2.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::ScaleAdd(a, 2.f, b, out);
    for (auto i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(4.f));
    }
}

TEST_CASE("ScaleAccumulate")
{
    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 1.f);
    std::vector<float> out(N, 0.f);

    ArrayMath::ScaleAccumulate(a, 2.f, out);
    for (auto i = 0; i < N; ++i)
    {
        CHECK(out[i] == doctest::Approx(2.f));
    }
}

TEST_SUITE_END();