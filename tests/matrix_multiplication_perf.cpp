#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

#include "rng.h"
#include <matrix_multiplication.h>
#include <sffdn/matrix_gallery.h>

#include <Eigen/Core>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

using namespace ankerl;
using namespace std::chrono_literals;

namespace
{
// clang-format off
alignas(64) constexpr std::array<float, 16 * 16> kMatrix16x16 = {
    0.8115,  0.7883,  -0.1009, -0.4798, 0.5561,  -0.8497, -0.4591, -0.5082, 0.2942,  -0.9076, 1.4608,  0.0249,  0.1279, 0.6496,  -0.7438, 0.6983,
    -0.6382, -0.5792, 1.8077,  -0.6436, -1.1647, 0.3072,  1.5258,  0.6047,  0.3062,  -0.5011, 0.1370,  -0.6525, -0.4378, 0.5765,  -0.2199, -1.5274,
    -0.8997, 0.3867,  0.9229,  2.4441,  1.0242,  1.7132,  1.4185, -0.2930, -1.1823, 0.7220,  0.3529,  0.3051,  0.2710,  -0.4333, -0.9739, 0.8087,
    0.5787,  -0.8851, 0.8756,  -0.9036, -0.9290, 0.6605,  1.6982,  -0.7414, 0.1653,  -0.1662, -0.5312, -0.0468, 2.0139,  1.1551,  2.2702,  1.0435,
    -2.4733, 0.2897,  0.6500,  0.4321,  -0.9941, 0.4259,  0.3659,  0.6063,  1.0566,  1.2516,  0.4336,  1.2364,  -0.0745, 2.1601, -0.4327, 1.5009,
    0.5420,  -0.4444, -0.6327, 0.9789,  0.0477,  -0.2813, -1.1436, -1.1438, -1.5221, -1.3656, -0.0268, -1.8896, 0.5142,  1.6477,  0.5834,  1.3712,
    0.7138,  -1.0938, 0.9510,  0.2010,  -0.3050, 0.4406,  -1.0982, 1.1552, -1.4084, 0.4542,  0.4645,  1.9505,  -0.2926, -0.1505, -0.4178, -1.4720,
    0.3553,  0.5111,  0.7077,  0.5046,  -0.0559, -0.0640, -0.1741, -0.6415, 0.5213,  -0.5516, 0.6321,  0.4983,  0.0716,  0.7190,  1.2294,  0.8492,
    -0.4372, 0.0871, -0.6325, 1.5058,  -1.9543, 0.3722,  -0.1877, -0.6004, 1.0912,  1.6989,  1.4868,  0.6140,  1.7396,  0.8925,  0.8093, 0.0768,
    -0.3793, 0.6539,  0.2240,  0.7582,  1.2189,  0.7684,  0.0411,  1.0480,  0.0246,  0.4290,  -0.0928, 0.0911, 0.4818,  -0.4226, -0.2805, -0.5073,
    1.2705,  -0.4315, 0.3078,  0.5731,  -1.6614, -0.5538, -1.4189, -0.3264, -0.9412, 0.3729,  0.9967,  -1.0853, 0.4299,  1.4532,  -1.3039, -0.3152,
    1.7364,  -1.6748, -0.1443, 1.4996,  0.5037,  0.6728, -0.1052, 0.8427,  0.9272,  -1.9327, 0.9896,  -1.9199, 1.9027,  0.0582,  -0.0846, 1.3085,
    0.1397,  0.2934,  0.3757, 1.0358,  0.6032,  0.1678,  1.1164,  -1.7259, -0.1818, -0.7826, 0.7863,  -0.1727, -1.1141, 0.5069,  -1.1348, 0.6243,
    1.1579,  -0.1751, -0.9790, 0.0794,  1.9442,  -0.1745, 0.5733,  0.5105,  -0.8059, 0.4847,  1.6786,  1.2738,  0.6281, 1.8806,  0.1477,  1.2901,
    -0.8213, 1.0182,  0.7621,  0.0256,  -0.3537, -0.8841, 0.8080,  1.2455,  -1.9410, -0.4243, 1.0317,  -0.0412, 1.6680,  -0.4045, -0.4402, 0.1076,
    -0.6334, 0.2719,  -0.1574, -0.4491, 0.9554,  1.1533,  -0.0829, -0.4150, 0.1559,  -0.8750, 2.8035,  0.9523,  -1.0604, -0.3215, 1.3143,  0.6825,
};

//clang-format on

constexpr std::array<float, 16> kInput = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

std::array<float, 16> ScalarMatrixMultiply16()
{
    std::array<float, 16> output{};
    for (auto output_index = 0u; output_index < output.size(); ++output_index)
    {
        for (auto input_index = 0u; input_index < kInput.size(); ++input_index)
        {
            output[output_index] += kMatrix16x16[(output_index * kInput.size()) + input_index] * kInput[input_index];
        }
    }
    return output;
}

void RequireClose(const std::array<float, 16>& expected, const std::array<float, 16>& actual)
{
    for (auto output = 0u; output < expected.size(); ++output)
    {
        REQUIRE_THAT(actual[output], Catch::Matchers::WithinAbs(expected[output], 1e-4f));
    }
}

} // namespace

TEST_CASE("MatrixMultiplicationPerf_single")
{
    constexpr uint32_t kMatSize = 16;
    const auto expected_output = ScalarMatrixMultiply16();
    std::array<float, kMatSize> validation_output{};
    sfFDN::MatrixMultiply_C(kInput, validation_output, kMatrix16x16, kMatSize);
    RequireClose(expected_output, validation_output);
    sfFDN::MatrixMultiply_16(kInput, validation_output, kMatrix16x16);
    RequireClose(expected_output, validation_output);

    alignas(64) std::array<float, kMatSize> eigen_output_data{};
    Eigen::Map<Eigen::Matrix<float, 1, kMatSize>> eigen_output(eigen_output_data.data());

    nanobench::Bench bench;
    bench.title("Matrix Multiplication Performance");
    // bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochIterations(1000000);

    {
        Eigen::Map<const Eigen::Matrix<float, kMatSize, kMatSize>> mat(kMatrix16x16.data());
        Eigen::Map<const Eigen::Matrix<float, 1, kMatSize>> input(kInput.data());
        Eigen::Map<Eigen::Matrix<float, 1, kMatSize>> output_map(eigen_output_data.data());
        bench.run("Eigen", [&]() {
            output_map.noalias() = input * mat;
            nanobench::doNotOptimizeAway(output_map);
        });
    }

    {

        Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned64> mat(kMatrix16x16.data(), kMatSize, kMatSize);
        Eigen::Map<const Eigen::RowVectorXf, Eigen::Aligned64> input(kInput.data(), kMatSize);
        Eigen::Map<Eigen::RowVectorXf, Eigen::Aligned64> output_map(eigen_output_data.data(), kMatSize);
        bench.run("Eigen_Dynamic", [&]() {
            output_map.noalias() = input * mat;
            nanobench::doNotOptimizeAway(output_map);
        });
    }

    std::array<float, kMatSize> output{};
    bench.run("MatrixMultiply", [&]() {
        sfFDN::MatrixMultiply_C(kInput, output, kMatrix16x16, 16);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("MatrixMultiply_unroll", [&]() {
        sfFDN::MatrixMultiply_16(kInput, output, kMatrix16x16);
        nanobench::doNotOptimizeAway(output);
    });

    #ifdef __APPLE__
    bench.run("vDSP", [&]() {
        const float* A = kInput.data();
        const float* B = kMatrix16x16.data();
        float* C = output.data();
        vDSP_mmul(A, 1, B, 1, C, 1, 1, 16, 16);
        nanobench::doNotOptimizeAway(output);
    });
    #endif
}

TEST_CASE("MatrixMultiplicationPerf_block")
{
    constexpr uint32_t kMatSize = 16;
    constexpr uint32_t kBlockSize = 128;

    constexpr uint32_t kInputSize = kMatSize * kBlockSize;

    std::array<float, kInputSize> input{};
    for (auto i = 0u; i < kMatSize; ++i)
    {
        for (auto j = 0u; j < kBlockSize; ++j)
        {
            input[(i * kBlockSize) + j] = kInput[i];
        }
    }

    nanobench::Bench bench;
    bench.title("Matrix Multiplication Performance - Block");
    // bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochIterations(50000);
    // bench.batch(2*kInputSize * kMatSize * kMatSize);
    bench.performanceCounters(true);

    Eigen::Map<const Eigen::MatrixXf> mat(kMatrix16x16.data(), kMatSize, kMatSize);
    Eigen::Map<const Eigen::MatrixXf> eigen_input(input.data(), kBlockSize, kMatSize);
    std::array<float, kInputSize> output{};
    bench.run("Eigen", [&]() {

        Eigen::Map<Eigen::MatrixXf> output_eigen(output.data(), kBlockSize, kMatSize);

        output_eigen.noalias() = eigen_input * mat;
        nanobench::doNotOptimizeAway(output_eigen);
    });


    Eigen::Matrix<float, kMatSize, kMatSize> eigen_mat_static = mat;
    Eigen::Matrix<float, kBlockSize, kMatSize> eigen_input_static = eigen_input;
    Eigen::Matrix<float, kBlockSize, kMatSize> eigen_output_static;

    bench.run("EigenStatic", [&]() {
        eigen_output_static.noalias() = eigen_input_static * eigen_mat_static;
        nanobench::doNotOptimizeAway(eigen_output_static);
    });


    #ifdef __APPLE__
    bench.minEpochIterations(200000);
    bench.run("vDSP", [&]() {
        const float* A = input.data();
        const float* B = kMatrix16x16.data();
        float* C = output.data();
        vDSP_mmul(B, 1, A, 1, C, 1, 16, kBlockSize, 16);
        nanobench::doNotOptimizeAway(C);
        nanobench::doNotOptimizeAway(A);
        nanobench::doNotOptimizeAway(B);
    });
    #endif
}

TEST_CASE("Hadamard")
{
    constexpr uint32_t kMatSize = 8;
    std::array<float, kMatSize> input{};
    std::iota(input.begin(), input.end(), 1.f);

    auto hadamard = sfFDN::GenerateMatrix(kMatSize, sfFDN::ScalarMatrixType::Hadamard);

    constexpr uint32_t kIterations = 1000;
    nanobench::Bench bench;
    bench.title("Hadamard Multiplication Performance");
    // bench.timeUnit(1us, "us");
    bench.batch(kMatSize * kIterations);
    bench.relative(true);
    bench.minEpochIterations(1000);

    Eigen::Map<const Eigen::MatrixXf> eigen_mat(hadamard.data(), kMatSize, kMatSize);
    Eigen::Map<const Eigen::RowVectorXf> eigen_input(input.data(), kMatSize);
    bench.run("Eigen", [&]() {

        std::array<float, kMatSize> eigen_output_data{};
        Eigen::Map<Eigen::RowVectorXf> eigen_output(eigen_output_data.data(), kMatSize);

        for (auto i = 0u; i < kIterations; ++i)
        {
            eigen_output.noalias() = eigen_input * eigen_mat;
            nanobench::doNotOptimizeAway(eigen_output);
        }
    });

    std::array<float, kMatSize> output{};

    bench.minEpochIterations(1000);
    bench.run("MatrixMultiply", [&]() {
        for (auto i = 0u; i < kIterations; ++i)
        {
            sfFDN::MatrixMultiply_C(input, output, hadamard, kMatSize);
            nanobench::doNotOptimizeAway(output);
        }
    });

    bench.minEpochIterations(1000);
    bench.run("HadamardMultiply", [&]() {
        for (auto i = 0u; i < kIterations; ++i)
        {
            sfFDN::HadamardMultiply(input, output);
            nanobench::doNotOptimizeAway(output);
        }
    });

    bench.run("WalshHadamardTransform", [&]() {
        std::array<float, kMatSize> inout{};
        std::ranges::copy(input, inout.begin());
        for (auto i = 0u; i < kIterations; ++i)
        {
            sfFDN::WalshHadamardTransform(inout);
            nanobench::doNotOptimizeAway(inout);
        }
    });

    bench.run("FWHT", [&]() {
        std::array<float, kMatSize> inout{};
        std::ranges::copy(input, inout.begin());
        for (auto i = 0u; i < kIterations; ++i)
        {
            sfFDN::FWHT<kMatSize>(inout);
            nanobench::doNotOptimizeAway(inout);
        }
    });
}

TEST_CASE("Hadamard_Block")
{
    constexpr uint32_t kMatSize = 16;
    constexpr uint32_t kBlockSize = 1;
    std::array<float, kMatSize * kBlockSize> input{};
    sfFDN::RNG rng;
    for (auto& i : input)
    {
        i = rng();
    }

    auto hadamard = sfFDN::GenerateMatrix(kMatSize, sfFDN::ScalarMatrixType::Hadamard);

    nanobench::Bench bench;
    bench.title("Hadamard Multiplication Performance");
    // bench.timeUnit(1us, "us");
    // bench.batch(kMatSize * kIterations);
    bench.relative(true);
    bench.minEpochIterations(10000);


    Eigen::Map<const Eigen::MatrixXf> eigen_mat(hadamard.data(), kMatSize, kMatSize);
    Eigen::Map<const Eigen::MatrixXf> eigen_input(input.data(), kBlockSize, kMatSize);

    std::array<float, kMatSize * kBlockSize> eigen_output_data{};
    Eigen::Map<Eigen::MatrixXf> eigen_output(eigen_output_data.data(), kBlockSize, kMatSize);
    bench.run("Eigen", [&]() {
            eigen_output.noalias() = eigen_input * eigen_mat;
            nanobench::doNotOptimizeAway(eigen_output);
    });

    std::array<float, kMatSize * kBlockSize> output{};

    bench.run("MatrixMultiply", [&]() {
        sfFDN::MatrixMultiply_C(input, output, hadamard, kMatSize);
        nanobench::doNotOptimizeAway(output);
    });

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }

    bench.run("HadamardMultiply", [&]() {
        std::array<float, kMatSize> input_block{};
        std::array<float, kMatSize> output_block{};
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            for (auto j = 0u; j < kMatSize; ++j)
            {
                input_block[j] = input[i + j * kBlockSize];
            }
            sfFDN::HadamardMultiply(input_block, output_block);

            for (auto j = 0u; j < kMatSize; ++j)
            {
                output[i + j * kBlockSize] = output_block[j];
            }
        }
        nanobench::doNotOptimizeAway(output);
    });

    // Check results
    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-4));
    }

    bench.run("WalshHadamardTransform", [&]() {
        std::array<float, kMatSize> inout{};
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            for (auto j = 0u; j < kMatSize; ++j)
            {
                inout[j] = input[i + j * kBlockSize];
            }
            sfFDN::WalshHadamardTransform(inout);
            for (auto j = 0u; j < kMatSize; ++j)
            {
                output[i + j * kBlockSize] = inout[j];
            }
        }
        nanobench::doNotOptimizeAway(output);
    });

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }

    bench.run("FWHT", [&]() {
        std::array<float, kMatSize> inout{};
        for (auto i = 0u; i < kBlockSize; ++i)
        {
            for (auto j = 0u; j < kMatSize; ++j)
            {
                inout[j] = input[i + j * kBlockSize];
            }
            sfFDN::FWHT<kMatSize>(inout);
            for (auto j = 0u; j < kMatSize; ++j)
            {
                output[i + j * kBlockSize] = inout[j];
            }
        }
        nanobench::doNotOptimizeAway(output);
        nanobench::doNotOptimizeAway(inout);
    });

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }

    bench.run("FWHT_withTranspose", [&]() {
        // Transpose the (kBlockSize x kMatSize) input into a scratch buffer so that
        // each block's kMatSize-vector becomes contiguous (kMatSize x kBlockSize).
        std::array<float, kMatSize * kBlockSize> scratch{};
        Eigen::Map<Eigen::MatrixXf> scratch_map(scratch.data(), kMatSize, kBlockSize);
        scratch_map.noalias() = eigen_input.transpose();

        for (auto i = 0u; i < kBlockSize; ++i)
        {
            auto inout = std::span(scratch).subspan(i * kMatSize, kMatSize);
            sfFDN::FWHT<kMatSize>(std::span<float, kMatSize>(inout));
        }

        // Transpose back into the expected (kBlockSize x kMatSize) output layout.
        Eigen::Map<Eigen::MatrixXf> output_map(output.data(), kBlockSize, kMatSize);
        output_map.noalias() = scratch_map.transpose();
        nanobench::doNotOptimizeAway(output);
    });

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }

    bench.run("FWHT_Batched", [&]() {
        // The input is (kBlockSize x kMatSize) column-major, so column n holds the
        // n-th element of every block contiguously. The FWHT butterflies therefore
        // operate column-vs-column across all blocks at once (vectorized), with no
        // transpose required.
        Eigen::Map<Eigen::Matrix<float, kBlockSize, kMatSize>> mat(output.data());
        mat = eigen_input;

        Eigen::Matrix<float, kBlockSize, 1> a;
        for (auto h = 1u; h < kMatSize; h *= 2)
        {
            for (auto i = 0u; i < kMatSize; i += 2 * h)
            {
                for (auto j = 0u; j < h; ++j)
                {
                    a = mat.col(i + j);
                    mat.col(i + j) = a + mat.col(i + j + h);
                    mat.col(i + j + h) = a - mat.col(i + j + h);
                }
            }
        }

        mat *= 1.f / std::sqrt(static_cast<float>(kMatSize));
        nanobench::doNotOptimizeAway(output);
    });

    for (auto i = 0u; i < kBlockSize * kMatSize; ++i)
    {
        REQUIRE_THAT(eigen_output_data[i], Catch::Matchers::WithinAbs(output[i], 1e-5));
    }
}
