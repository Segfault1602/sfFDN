#include "doctest.h"

#include <array>
#include <iostream>
#include <span>
#include <vector>

#include "sffdn/sffdn.h"

#include "matrix_multiplication.h"

#include <Eigen/Core>

namespace
{
template <size_t N>
void TestMatrixMultiply_Eye()
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
    sfFDN::MatrixMultiply_C(input, output, matrix_span, N);

    for (size_t i = 0; i < N; ++i)
    {
        CHECK(input[i] == doctest::Approx(output[i]));
    }
}
} // namespace

TEST_SUITE_BEGIN("MatrixMultiplication");

TEST_CASE("Identity")
{
    TestMatrixMultiply_Eye<4>();
    TestMatrixMultiply_Eye<8>();
    TestMatrixMultiply_Eye<16>();
    TestMatrixMultiply_Eye<32>();
}

TEST_CASE("MatrixMultiply")
{
    constexpr std::array kNSize = {4, 8, 10, 12, 16, 32};
    constexpr std::array kRowCounts = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};

    for (auto N : kNSize)
    {
        for (auto kRowCount : kRowCounts)
        {
            const size_t kInputSize = N * kRowCount;

            std::vector<float> input(kInputSize);
            for (size_t i = 0; i < kInputSize; ++i)
            {
                input[i] = rand() % 1000 / 1000.f;
            }

            std::vector<float> matrix = sfFDN::GenerateMatrix(N, sfFDN::ScalarMatrixType::Random, 123);

            std::vector<float> expected_output(kInputSize, 0.f);

            Eigen::Map<Eigen::MatrixXf> input_map(input.data(), kRowCount, N);
            Eigen::Map<Eigen::MatrixXf> matrix_map(matrix.data(), N, N);
            Eigen::Map<Eigen::MatrixXf> expected_output_map(expected_output.data(), kRowCount, N);
            expected_output_map = input_map * matrix_map;

            std::vector<float> output(kInputSize, 0.f);
            sfFDN::MatrixMultiply_C(input, output, matrix, N);

            for (size_t i = 0; i < output.size(); ++i)
            {
                CHECK(expected_output[i] == doctest::Approx(output[i]));
            }
        }
    }
}

TEST_CASE("MatrixMultiply_6")
{
    constexpr size_t N = 6;
    constexpr size_t kRowCount = 4;
    constexpr size_t kInputSize = N * kRowCount;

    // clang-format off
    float a[kInputSize] = {0.4889,  0.2939,  -1.0689, 0.3252,
                          -0.1022, -0.8649, 1.0347,  -0.7873,
                          -0.8095, -0.7549, -0.2414, -0.0301,
                           0.7269,  0.8884,  -2.9443, 1.3703,
                           0.3192,  -0.1649, -0.3034, -1.1471,
                            1.4384,  -1.7115, 0.3129,  0.6277};
    std::array<float, N * N> matrix = {
        1.0933,  1.1093, -0.8637, 0.0774,  -1.2141, -1.1135,
       -0.0068, 1.5326,  -0.7697, 0.3714, -0.2256, 1.1174,
       -1.0891, 0.0326, 0.5525,  1.1006,  1.5442,  0.0859,
       -1.4916, -0.7423, -1.0616, 2.3505, -0.6156, 0.7481,
       -0.1924, 0.8886, -0.7648, -1.4023, -1.4224, 0.4882,
       -0.1774, -0.1961, 1.4193,  0.2916, 0.1978,  1.5877,
    };
    // clang-format on

    float output[kInputSize] = {0.f};

    auto matrix_span = std::span<const float, N * N>(matrix.data(), N * N);
    sfFDN::MatrixMultiply_C(a, output, matrix_span, N);

    Eigen::Map<Eigen::Matrix<float, kRowCount, N>> input_map(a);
    Eigen::Map<Eigen::Matrix<float, N, N>> matrix_map(matrix.data());

    Eigen::Matrix<float, kRowCount, N> expected = input_map * matrix_map;

    for (size_t i = 0; i < kRowCount; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            CHECK(expected(i, j) == doctest::Approx(output[i + j * kRowCount]));
        }
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

        sfFDN::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        sfFDN::WalshHadamardTransform(input);
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

        sfFDN::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        sfFDN::WalshHadamardTransform(input);
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

        sfFDN::HadamardMultiply(input, output);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(output[i]));
        }

        sfFDN::WalshHadamardTransform(input);
        for (size_t i = 0; i < input.size(); i += N)
        {
            CHECK(expected[i] == doctest::Approx(input[i]));
        }
    }
}

TEST_SUITE_END();