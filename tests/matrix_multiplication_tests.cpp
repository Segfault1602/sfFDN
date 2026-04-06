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
    constexpr std::array<float, kInputSize> kInput = {0.4889f,  0.2939f,  -1.0689f, 0.3252f,
                          -0.1022f, -0.8649f, 1.0347f,  -0.7873f,
                          -0.8095f, -0.7549f, -0.2414f, -0.0301f,
                           0.7269f,  0.8884f,  -2.9443f, 1.3703f,
                           0.3192f,  -0.1649f, -0.3034f, -1.1471f,
                            1.4384f,  -1.7115f, 0.3129f,  0.6277f};

    constexpr std::array<float, kMatSize * kMatSize> kMatrix = {
        1.0933f,  1.1093f, -0.8637f, 0.0774f,  -1.2141f, -1.1135f,
       -0.0068f, 1.5326f,  -0.7697f, 0.3714f, -0.2256f, 1.1174f,
       -1.0891f, 0.0326f, 0.5525f,  1.1006f,  1.5442f,  0.0859f,
       -1.4916f, -0.7423f, -1.0616f, 2.3505f, -0.6156f, 0.7481f,
       -0.1924f, 0.8886f, -0.7648f, -1.4023f, -1.4224f, 0.4882f,
       -0.1774f, -0.1961f, 1.4193f,  0.2916f, 0.1978f,  1.5877f,
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

TEST_CASE("MatrixMultiply_Hadamard")
{
    TestMatrixMultiplyHadamard<4>();
    TestMatrixMultiplyHadamard<8>();
    TestMatrixMultiplyHadamard<16>();
}

void FastWalshHadamardTransform_4(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output)
{
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == 4);

    Eigen::Map<const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> in(input.Data(), 4, input.SampleCount());
    Eigen::Map<Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> out(output.Data(), 4, output.SampleCount());

    out.row(0) = in.row(0) + in.row(1) + in.row(2) + in.row(3);
    out.row(1) = in.row(0) - in.row(1) + in.row(2) - in.row(3);
    out.row(2) = in.row(0) + in.row(1) - in.row(2) - in.row(3);
    out.row(3) = in.row(0) - in.row(1) - in.row(2) + in.row(3);

    out *= 0.5f;
}

TEST_CASE("Hadamard_4")
{
    constexpr uint32_t kMatSize = 4;
    constexpr uint32_t kBlockSize = 2;

    std::array<float, kMatSize * kBlockSize> input{};
    sfFDN::RNG rng;
    for (auto& i : input)
    {
        i = rng();
    }

    auto hadamard = sfFDN::GenerateMatrix(kMatSize, sfFDN::ScalarMatrixType::Hadamard);
    Eigen::Map<const Eigen::MatrixXf> eigen_mat(hadamard.data(), kMatSize, kMatSize);
    Eigen::Map<const Eigen::MatrixXf> eigen_input(input.data(), kBlockSize, kMatSize);

    std::array<float, kMatSize * kBlockSize> eigen_output_data{};
    Eigen::Map<Eigen::MatrixXf> eigen_output(eigen_output_data.data(), kBlockSize, kMatSize);
    eigen_output.noalias() = eigen_input * eigen_mat;

    std::array<float, kMatSize * kBlockSize> output{};
    sfFDN::AudioBuffer input_buffer(kBlockSize, kMatSize, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kMatSize, output);

    FastWalshHadamardTransform_4(input_buffer, output_buffer);

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }
}