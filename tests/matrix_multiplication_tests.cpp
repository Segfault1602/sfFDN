#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "rng.h"
#include "sffdn/sffdn.h"

#include "matrix_gallery_internal.h"
#include "matrix_multiplication.h"

#include <Eigen/Core>

namespace
{
constexpr uint32_t kMatrixMultiplyOrder = 16;

std::array<float, kMatrixMultiplyOrder * kMatrixMultiplyOrder> MakeNonSymmetricMatrix()
{
    std::array<float, kMatrixMultiplyOrder * kMatrixMultiplyOrder> matrix{};
    for (auto output = 0u; output < kMatrixMultiplyOrder; ++output)
    {
        for (auto input = 0u; input < kMatrixMultiplyOrder; ++input)
        {
            matrix[(output * kMatrixMultiplyOrder) + input] =
                static_cast<float>((17 * output) - (5 * input) + (3 * output * input) + 1) / 11.f;
        }
    }
    return matrix;
}

std::array<float, kMatrixMultiplyOrder> ScalarMatrixMultiply(
    const std::array<float, kMatrixMultiplyOrder>& input,
    const std::array<float, kMatrixMultiplyOrder * kMatrixMultiplyOrder>& matrix)
{
    std::array<float, kMatrixMultiplyOrder> output{};
    for (auto output_index = 0u; output_index < kMatrixMultiplyOrder; ++output_index)
    {
        for (auto input_index = 0u; input_index < kMatrixMultiplyOrder; ++input_index)
        {
            output[output_index] += matrix[(output_index * kMatrixMultiplyOrder) + input_index] * input[input_index];
        }
    }
    return output;
}

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

TEST_CASE("MatrixMultiply_16 matches MatrixMultiply_C for a row-major order-16 matrix", "[matrix_multiplication]")
{
    const std::array<float, kMatrixMultiplyOrder> input = {1.f,  -2.f,  3.5f, -4.f,   5.f,  -6.5f, 7.f,   -8.f,
                                                           9.5f, -10.f, 11.f, -12.5f, 13.f, -14.f, 15.5f, -16.f};
    const auto matrix = MakeNonSymmetricMatrix();
    const auto expected = ScalarMatrixMultiply(input, matrix);

    std::array<float, kMatrixMultiplyOrder> c_output{};
    std::array<float, kMatrixMultiplyOrder> unrolled_output{};
    sfFDN::MatrixMultiply_C(input, c_output, matrix, kMatrixMultiplyOrder);
    sfFDN::MatrixMultiply_16(input, unrolled_output, matrix);

    for (auto output = 0u; output < kMatrixMultiplyOrder; ++output)
    {
        REQUIRE_THAT(c_output[output], Catch::Matchers::WithinAbs(expected[output], 1e-4f));
        REQUIRE_THAT(unrolled_output[output], Catch::Matchers::WithinAbs(expected[output], 1e-4f));
    }
}

TEST_CASE("MatrixMultiply_C preserves identity matrices across supported orders", "[matrix_multiplication]")
{
    TestMatrixMultiplyIdentity<4>();
    TestMatrixMultiplyIdentity<8>();
    TestMatrixMultiplyIdentity<16>();
    TestMatrixMultiplyIdentity<32>();
}

TEST_CASE("MatrixMultiply_C matches Eigen for multiple orders and row counts", "[matrix_multiplication]")
{
    constexpr std::array kNSize = {4, 6, 8, 10, 12, 16, 32};
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

template <size_t N>
void TestMatrixMultiplyHadamard()
{
    auto eigen_mat = sfFDN::HadamardMatrix(N);
    Eigen::RowVectorXf eigen_input;
    eigen_input.resize(N);
    std::array<float, N> input;
    for (auto i = 0u; i < N; ++i)
    {
        eigen_input(i) = static_cast<float>(i + 1);
        input[i] = static_cast<float>(i + 1);
    }

    std::array<float, N> output{};

    auto eigen_output = eigen_input * eigen_mat;

    sfFDN::HadamardMultiply(input, output);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(eigen_output(i), 1e-5));
    }

    sfFDN::WalshHadamardTransform(input);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(input[i], Catch::Matchers::WithinAbs(eigen_output(i), 1e-5));
    }
}

TEST_CASE("HadamardMultiply matches WalshHadamardTransform against Hadamard matrices", "[matrix_multiplication]")
{
    TestMatrixMultiplyHadamard<4>();
    TestMatrixMultiplyHadamard<8>();
    TestMatrixMultiplyHadamard<16>();
}

TEST_CASE("HadamardMultiplyBlock applies a Hadamard transform to each block sample", "[matrix_multiplication]")
{
    constexpr uint32_t kMatrixSize = 4;
    constexpr uint32_t kBlockSize = 5;

    std::array<float, kMatrixSize * kBlockSize> input{};
    for (auto channel = 0u; channel < kMatrixSize; ++channel)
    {
        for (auto sample = 0u; sample < kBlockSize; ++sample)
        {
            input[(channel * kBlockSize) + sample] = static_cast<float>((10 * channel) + sample + 1);
        }
    }

    std::array<float, kMatrixSize * kBlockSize> expected{};
    for (auto sample = 0u; sample < kBlockSize; ++sample)
    {
        std::array<float, kMatrixSize> values{};
        for (auto channel = 0u; channel < kMatrixSize; ++channel)
        {
            values[channel] = input[(channel * kBlockSize) + sample];
        }
        expected[sample] = (values[0] + values[1] + values[2] + values[3]) * 0.5f;
        expected[kBlockSize + sample] = (values[0] - values[1] + values[2] - values[3]) * 0.5f;
        expected[(2 * kBlockSize) + sample] = (values[0] + values[1] - values[2] - values[3]) * 0.5f;
        expected[(3 * kBlockSize) + sample] = (values[0] - values[1] - values[2] + values[3]) * 0.5f;
    }

    std::array<float, kMatrixSize * kBlockSize> output{};
    sfFDN::AudioBuffer const input_buffer(kBlockSize, kMatrixSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatrixSize, output);
    sfFDN::HadamardMultiplyBlock(input_buffer, output_buffer);

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 1e-5f));
    }
}