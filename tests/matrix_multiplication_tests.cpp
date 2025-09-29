#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "rng.h"
#include "sffdn/sffdn.h"

#include "matrix_multiplication.h"

#include <Eigen/Core>

namespace
{
template <uint32_t N>
void TestMatrixMultiplyIdentity()
{
    std::vector<float> matrix(N * N, 0.f);
    for (auto i = 0u; i < N; ++i)
    {
        matrix[i + i * N] = 1.f;
    }

    std::vector<float> input(N, 0.f);
    sfFDN::RNG rng;
    for (auto i = 0u; i < N; ++i)
    {
        input[i] = rng();
    }

    std::vector<float> output(N, 0.f);

    sfFDN::MatrixMultiply_C(input, output, matrix, N);

    for (auto i = 0u; i < N; ++i)
    {
        REQUIRE_THAT(input[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
    }
}
} // namespace

TEST_CASE("Identity")
{
    TestMatrixMultiplyIdentity<4>();
    TestMatrixMultiplyIdentity<8>();
    TestMatrixMultiplyIdentity<16>();
    TestMatrixMultiplyIdentity<32>();
}

TEST_CASE("MatrixMultiply")
{
    constexpr std::array kNSize = {4, 8, 10, 12, 16, 32};
    constexpr std::array kRowCounts = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};

    sfFDN::RNG rng;
    for (auto mat_size : kNSize)
    {
        for (auto row_count : kRowCounts)
        {
            const uint32_t input_size = mat_size * row_count;

            std::vector<float> input(input_size);
            for (auto& i : input)
            {
                i = rng();
            }

            std::vector<float> matrix = sfFDN::GenerateMatrix(mat_size, sfFDN::ScalarMatrixType::Random, 123);

            std::vector<float> expected_output(input_size, 0.f);

            Eigen::Map<Eigen::MatrixXf> input_map(input.data(), row_count, mat_size);
            Eigen::Map<Eigen::MatrixXf> matrix_map(matrix.data(), mat_size, mat_size);
            Eigen::Map<Eigen::MatrixXf> expected_output_map(expected_output.data(), row_count, mat_size);
            expected_output_map = input_map * matrix_map;

            std::vector<float> output(input_size, 0.f);
            sfFDN::MatrixMultiply_C(input, output, matrix, mat_size);

            for (auto i = 0u; i < output.size(); ++i)
            {
                REQUIRE_THAT(expected_output[i], Catch::Matchers::WithinAbs(output[i], 1e-6));
            }
        }
    }
}

TEST_CASE("MatrixMultiply_6")
{
    constexpr uint32_t kMatSize = 6;
    constexpr uint32_t kRowCount = 4;
    constexpr uint32_t kInputSize = kMatSize * kRowCount;

    // clang-format off
    constexpr std::array<float, kInputSize> kInput = {0.4889,  0.2939,  -1.0689, 0.3252,
                          -0.1022, -0.8649, 1.0347,  -0.7873,
                          -0.8095, -0.7549, -0.2414, -0.0301,
                           0.7269,  0.8884,  -2.9443, 1.3703,
                           0.3192,  -0.1649, -0.3034, -1.1471,
                            1.4384,  -1.7115, 0.3129,  0.6277};

    constexpr std::array<float, kMatSize * kMatSize> kMatrix = {
        1.0933,  1.1093, -0.8637, 0.0774,  -1.2141, -1.1135,
       -0.0068, 1.5326,  -0.7697, 0.3714, -0.2256, 1.1174,
       -1.0891, 0.0326, 0.5525,  1.1006,  1.5442,  0.0859,
       -1.4916, -0.7423, -1.0616, 2.3505, -0.6156, 0.7481,
       -0.1924, 0.8886, -0.7648, -1.4023, -1.4224, 0.4882,
       -0.1774, -0.1961, 1.4193,  0.2916, 0.1978,  1.5877,
    };
    // clang-format on

    std::array<float, kInputSize> output = {0.f};

    sfFDN::MatrixMultiply_C(kInput, output, kMatrix, kMatSize);

    Eigen::Map<const Eigen::Matrix<float, kRowCount, kMatSize>> input_map(kInput.data());
    Eigen::Map<const Eigen::Matrix<float, kMatSize, kMatSize>> matrix_map(kMatrix.data());

    Eigen::Matrix<float, kRowCount, kMatSize> expected = input_map * matrix_map;

    for (auto i = 0u; i < kRowCount; ++i)
    {
        for (auto j = 0u; j < kMatSize; ++j)
        {
            REQUIRE_THAT(expected(i, j),
                         Catch::Matchers::WithinAbs(output[i + j * kRowCount], std::numeric_limits<float>::epsilon()));
        }
    }
}

TEST_CASE("Hadamard")
{
    SECTION("Hadamard_4")
    {
        constexpr uint32_t kMatSize = 4;

        std::array<float, kMatSize> input = {1, 2, 3, 4};
        std::array<float, kMatSize> output{};
        constexpr std::array<float, kMatSize> kExpected = {5, -1, -2, 0};

        sfFDN::HadamardMultiply(input, output);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }

        sfFDN::WalshHadamardTransform(input);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(input[i], std::numeric_limits<float>::epsilon()));
        }
    }

    SECTION("Hadamard_8")
    {
        constexpr uint32_t kMatSize = 8;

        std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, kMatSize> output{};

        constexpr std::array<float, kMatSize> kExpected = {
            12.727922061357855, -1.414213562373095, -2.828427124746190, 0, -5.656854249492380, 0, 0, 0};

        sfFDN::HadamardMultiply(input, output);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], 1e-6));
        }

        sfFDN::WalshHadamardTransform(input);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(input[i], 1e-6));
        }
    }

    SECTION("Hadamard_16")
    {
        constexpr uint32_t kMatSize = 16;

        std::array<float, kMatSize> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::array<float, kMatSize> output{};

        constexpr std::array<float, kMatSize> kExpected = {34, -2, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0};

        sfFDN::HadamardMultiply(input, output);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(output[i], std::numeric_limits<float>::epsilon()));
        }

        sfFDN::WalshHadamardTransform(input);
        for (auto i = 0u; i < input.size(); i += kMatSize)
        {
            REQUIRE_THAT(kExpected[i], Catch::Matchers::WithinAbs(input[i], std::numeric_limits<float>::epsilon()));
        }
    }
}
