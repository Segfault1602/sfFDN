#include "doctest.h"

#include <array>
#include <iostream>
#include <span>
#include <vector>

#include "matrix_multiplication.h"

namespace
{
template <size_t N>
void TestMatrixMultipl_Eye()
{
    std::vector<float> matrix(N * N, 0.f);
    for (size_t i = 0; i < N; ++i)
    {
        matrix[i + i * N] = 1.f;
    }

    std::vector<float> input(N, 0.f);
    for (size_t i = 0; i < N; ++i)
    {
        input[i] = rand();
    }

    std::vector<float> output(N, 0.f);

    auto matrix_span = std::span<const float, N * N>(matrix.data(), N * N);
    fdn::MatrixMultiply(input, output, matrix_span, N);

    for (size_t i = 0; i < N; ++i)
    {
        CHECK(input[i] == doctest::Approx(output[i]));
    }
}
} // namespace

TEST_SUITE_BEGIN("MatrixMultiplication");

TEST_CASE("Identity")
{
    TestMatrixMultipl_Eye<4>();
    TestMatrixMultipl_Eye<8>();
    TestMatrixMultipl_Eye<16>();
    TestMatrixMultipl_Eye<32>();
}

TEST_CASE("MatrixMultiply_6")
{
    constexpr size_t N = 6;
    constexpr size_t kRowCount = 4;
    constexpr size_t kInputSize = N * kRowCount;

    float a[kInputSize] = {0.4889,  0.2939,  -1.0689, 0.3252,  -0.1022, -0.8649, 1.0347,  -0.7873,
                           -0.8095, -0.7549, -0.2414, -0.0301, 0.7269,  0.8884,  -2.9443, 1.3703,
                           0.3192,  -0.1649, -0.3034, -1.1471, 1.4384,  -1.7115, 0.3129,  0.6277};
    std::array<float, N * N> matrix = {
        1.0933,  1.1093, -0.8637, 0.0774,  -1.2141, -1.1135, -0.0068, 1.5326,  -0.7697, 0.3714, -0.2256, 1.1174,
        -1.0891, 0.0326, 0.5525,  1.1006,  1.5442,  0.0859,  -1.4916, -0.7423, -1.0616, 2.3505, -0.6156, 0.7481,
        -0.1924, 0.8886, -0.7648, -1.4023, -1.4224, 0.4882,  -0.1774, -0.1961, 1.4193,  0.2916, 0.1978,  1.5877,
    };

    float output[kInputSize] = {0.f};

    auto matrix_span = std::span<const float, N * N>(matrix.data(), N * N);
    fdn::MatrixMultiply(a, output, matrix_span, N);

    float expected[kInputSize] = {
        2.8961, 0.4472, -0.9876, 0.3676, 0.2517, -2.9600, 1.2252,  -0.8501, -2.8060, -1.7479, 1.1077, -1.4938,
        4.2254, 3.8755, -0.4025, 4.2830, 0.4453, -4.2811, -4.0578, -2.8680, -0.2588, -3.9689, 0.2004, 2.8797,
    };
    for (size_t i = 0; i < kInputSize; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected[i]).epsilon(0.0001f));
    }
}

TEST_CASE("Hadamard")
{
    SUBCASE("Hadamard_4")
    {
        constexpr size_t N = 4;

        std::array<float, N> input = {1, 2, 3, 4};
        std::array<float, N> output;
        constexpr std::array<float, N> expected = {5, -1, -2, 0};

        fdn::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        fdn::WalshHadamardTransform(input);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(input[i]));
        }
    }

    SUBCASE("Hadamard_8")
    {
        constexpr size_t N = 8;

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, N> output;

        constexpr std::array<float, N> expected = {
            12.727922061357855, -1.414213562373095, -2.828427124746190, 0, -5.656854249492380, 0, 0, 0};

        fdn::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        fdn::WalshHadamardTransform(input);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(input[i]));
        }
    }

    SUBCASE("Hadamard_16")
    {
        constexpr size_t N = 16;

        std::array<float, N> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, N> output;

        constexpr std::array<float, N> expected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        fdn::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        fdn::WalshHadamardTransform(input);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(input[i]));
        }
    }
}

TEST_SUITE_END();