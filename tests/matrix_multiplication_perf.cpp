#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>

#include <matrix_multiplication.h>

#include <Eigen/Core>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

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

constexpr std::array<float, 16*16> kMatrix16x16_T = {
    0.8115,   -0.6382,   -0.8997,    0.5787,   -2.4733,    0.5420,    0.7138,    0.3553,   -0.4372,   -0.3793,    1.2705,    1.7364,    0.1397,    1.1579,   -0.8213,   -0.6334,
    0.7883,   -0.5792,    0.3867,   -0.8851,    0.2897,   -0.4444,   -1.0938,    0.5111,    0.0871,    0.6539,   -0.4315,   -1.6748,    0.2934,   -0.1751,    1.0182,    0.2719,
   -0.1009,    1.8077,    0.9229,    0.8756,    0.6500,   -0.6327,    0.9510,    0.7077,   -0.6325,    0.2240,    0.3078,   -0.1443,    0.3757,   -0.9790,    0.7621,   -0.1574,
   -0.4798,   -0.6436,    2.4441,   -0.9036,    0.4321,    0.9789,    0.2010,    0.5046,    1.5058,    0.7582,    0.5731,    1.4996,    1.0358,    0.0794,    0.0256,   -0.4491,
    0.5561,   -1.1647,    1.0242,   -0.9290,   -0.9941,    0.0477,   -0.3050,   -0.0559,   -1.9543,    1.2189,   -1.6614,    0.5037,    0.6032,    1.9442,   -0.3537,    0.9554,
   -0.8497,    0.3072,    1.7132,    0.6605,    0.4259,   -0.2813,    0.4406,   -0.0640,    0.3722,    0.7684,   -0.5538,    0.6728,    0.1678,   -0.1745,   -0.8841,    1.1533,
   -0.4591,    1.5258,    1.4185,    1.6982,    0.3659,   -1.1436,   -1.0982,   -0.1741,   -0.1877,    0.0411,   -1.4189,   -0.1052,    1.1164,    0.5733,    0.8080,   -0.0829,
   -0.5082,    0.6047,   -0.2930,   -0.7414,    0.6063,   -1.1438,    1.1552,   -0.6415,   -0.6004,    1.0480,   -0.3264,    0.8427,   -1.7259,    0.5105,    1.2455,   -0.4150,
    0.2942,    0.3062,   -1.1823,    0.1653,    1.0566,   -1.5221,   -1.4084,    0.5213,    1.0912,    0.0246,   -0.9412,    0.9272,   -0.1818,   -0.8059,   -1.9410,    0.1559,
   -0.9076,   -0.5011,    0.7220,   -0.1662,    1.2516,   -1.3656,    0.4542,   -0.5516,    1.6989,    0.4290,    0.3729,   -1.9327,   -0.7826,    0.4847,   -0.4243,   -0.8750,
    1.4608,    0.1370,    0.3529,   -0.5312,    0.4336,   -0.0268,    0.4645,    0.6321,    1.4868,   -0.0928,    0.9967,    0.9896,    0.7863,    1.6786,    1.0317,    2.8035,
    0.0249,   -0.6525,    0.3051,   -0.0468,    1.2364,   -1.8896,    1.9505,    0.4983,    0.6140,    0.0911,   -1.0853,   -1.9199,   -0.1727,    1.2738,   -0.0412,    0.9523,
    0.1279,   -0.4378,    0.2710,    2.0139,   -0.0745,    0.5142,   -0.2926,    0.0716,    1.7396,    0.4818,    0.4299,    1.9027,   -1.1141,    0.6281,    1.6680,   -1.0604,
    0.6496,    0.5765,   -0.4333,    1.1551,    2.1601,    1.6477,   -0.1505,    0.7190,    0.8925,   -0.4226,    1.4532,    0.0582,    0.5069,    1.8806,   -0.4045,   -0.3215,
   -0.7438,   -0.2199,   -0.9739,    2.2702,   -0.4327,    0.5834,   -0.4178,    1.2294,    0.8093,   -0.2805,   -1.3039,   -0.0846,   -1.1348,    0.1477,   -0.4402,    1.3143,
    0.6983,   -1.5274,    0.8087,    1.0435,    1.5009,    1.3712,   -1.4720,    0.8492,    0.0768,   -0.5073,   -0.3152,    1.3085,    0.6243,    1.2901,    0.1076,    0.6825,
};
//clang-format on

alignas(64) constexpr std::array<float, 16> kInput = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
alignas(64) constexpr std::array<float, 16> kExpectedOutput = {11.2815,  -21.3455, 35.3144, 89.6171, 95.2097,  -6.8165,
                                                               -10.4919, 53.6382,  89.2180, 16.6164, -26.0199, 38.5943,
                                                               -13.2780, 103.6543, 12.4326, 52.7058};

bool IsAligned(const void* const Ptr, const size_t Align)
{
    // assume that 'Align' is valid (e.g. obtained from _Alignof )
    return (0 == (((size_t)(Ptr)) % Align));
}
} // namespace

TEST_CASE("MatrixMultiplicationPerf_single")
{
    constexpr uint32_t N = 16;

    if (IsAligned(kMatrix16x16.data(), 64))
    {
        std::cout << "Data is aligned correctly.\n";
    }
    else
    {
        std::cout << "Data is NOT aligned correctly.\n";
    }

    std::array<float, N> eigen_output_data{};
    Eigen::Map<Eigen::Matrix<float, 1, N>> eigen_output(eigen_output_data.data());

    nanobench::Bench bench;
    bench.title("Matrix Multiplication Performance");
    // bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochIterations(500000);

    {
        Eigen::Map<const Eigen::Matrix<float, N, N>> mat(kMatrix16x16.data());
        Eigen::Map<const Eigen::Matrix<float, 1, N>> input(kInput.data());
        Eigen::Map<Eigen::Matrix<float, 1, N>> output_map(eigen_output_data.data());
        bench.run("Eigen", [&]() {
            output_map.noalias() = input * mat;
            nanobench::doNotOptimizeAway(output_map);
        });
    }

    {

        Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned64> mat(kMatrix16x16.data(), N, N);
        Eigen::Map<const Eigen::RowVectorXf, Eigen::Aligned64> input(kInput.data(), N);
        Eigen::Map<Eigen::RowVectorXf, Eigen::Aligned64> output_map(eigen_output_data.data(), N);
        bench.run("Eigen_Dynamic", [&]() {
            output_map.noalias() = input * mat;
            nanobench::doNotOptimizeAway(output_map);
        });
    }

    std::array<float, N> output;
    bench.run("MatrixMultiply", [&]() {
        sfFDN::MatrixMultiply_C(kInput, output, kMatrix16x16, 16);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("MatrixMultiply_unroll", [&]() {
        sfFDN::MatrixMultiply_16(kInput, output, kMatrix16x16);
        nanobench::doNotOptimizeAway(output);
    });

    bench.run("vDSP", [&]() {
        const float* A = kInput.data();
        const float* B = kMatrix16x16.data();
        float* C = output.data();
        vDSP_mmul(A, 1, B, 1, C, 1, 1, 16, 16);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("MatrixMultiplicationPerf_block")
{
    constexpr uint32_t N = 16;
    constexpr uint32_t kBlockSize = 128;

    constexpr uint32_t kInputSize = N * kBlockSize;

    std::array<float, kInputSize> input;
    for (auto i = 0; i < N; ++i)
    {
        for (auto j = 0; j < kBlockSize; ++j)
        {
            input[i * kBlockSize + j] = kInput[i];
        }
    }

    nanobench::Bench bench;
    bench.title("Matrix Multiplication Performance - Block");
    // bench.timeUnit(1us, "us");
    bench.relative(true);
    bench.minEpochIterations(500000);
    bench.run("Eigen", [&]() {
        Eigen::Map<const Eigen::MatrixXf> mat(kMatrix16x16.data(), N, N);
        Eigen::Map<const Eigen::MatrixXf> eigen_input(input.data(), kBlockSize, N);


        std::array<float, kInputSize> eigen_output_data;
        Eigen::Map<Eigen::MatrixXf> output(eigen_output_data.data(), kBlockSize, N);

        output = eigen_input * mat;
        nanobench::doNotOptimizeAway(output);
    });

    std::array<float, kInputSize> output;
    bench.run("vDSP", [&]() {
        const float* A = input.data();
        const float* B = kMatrix16x16.data();
        float* C = output.data();
        vDSP_mmul(B, 1, A, 1, C, 1, 16, kBlockSize, 16);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("Hadamard")
{
    constexpr uint32_t N = 16;

    std::array<float, N * N> kHadamard = {
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1,
        1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, -1, 1,
        1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1, -1, -1, 1,  1,  1,  1,  -1, -1,
        -1, -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  -1, -1, -1, -1, 1,  1,
        1,  1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  1,
        1,  1,  1,  1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,
        -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, -1, 1,  1,  1,  -1, -1, 1,  1,  -1,
        -1, 1,  -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, 1,  1,  1,  1,
        1,  -1, 1,  -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, -1,
        1,  1,  1,  1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  -1, -1, 1,
    };

    for (auto i = 0; i < N * N; ++i)
    {
        kHadamard[i] *= 0.25f; // Scale down to avoid overflow in multiplication
    }

    // Eigen
    Eigen::Map<const Eigen::Matrix<float, N, N>> eigen_mat(kHadamard.data());
    Eigen::Map<const Eigen::Matrix<float, 1, N>> eigen_input(kInput.data());

    std::array<float, N> eigen_output_data;
    Eigen::Map<Eigen::Matrix<float, 1, N>> eigen_output(eigen_output_data.data());

    eigen_output = eigen_input * eigen_mat;

    std::cout << "Eigen Hadamard Output: ";
    std::cout << eigen_output << std::endl;

    std::array<float, N> output;
    sfFDN::MatrixMultiply_C(kInput, output, kHadamard, N);
    std::cout << "Custom Hadamard Output: ";
    for (const auto& val : output)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    sfFDN::HadamardMultiply(kInput, output);
    std::cout << "Hadamard Multiply Output: ";
    for (const auto& val : output)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::array<float, N> inout;
    for (auto i = 0; i < N; ++i)
    {
        inout[i] = kInput[i];
    }
    sfFDN::WalshHadamardTransform(inout);
    std::cout << "Walsh Hadamard Transform Output: ";
    for (const auto& val : inout)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    constexpr uint32_t kIterations = 1000;
    nanobench::Bench bench;
    bench.title("Hadamard Multiplication Performance");
    // bench.timeUnit(1us, "us");
    bench.batch(N * kIterations);
    bench.relative(true);
    bench.minEpochIterations(1000);

    bench.run("Eigen", [&]() {
        Eigen::Map<const Eigen::MatrixXf> eigen_mat(kHadamard.data(), N, N);
        Eigen::Map<const Eigen::RowVectorXf> eigen_input(kInput.data(), N);

        std::array<float, N> eigen_output_data;
        Eigen::Map<Eigen::RowVectorXf> eigen_output(eigen_output_data.data(), N);

        for (auto i = 0; i < kIterations; ++i)
        {
            eigen_output = eigen_input * eigen_mat;
            nanobench::doNotOptimizeAway(eigen_output);
        }
    });

    bench.run("MatrixMultiply", [&]() {
        std::array<float, N> output;
        for (auto i = 0; i < kIterations; ++i)
        {
            sfFDN::MatrixMultiply_C(kInput, output, kHadamard, N);
            nanobench::doNotOptimizeAway(output);
        }
    });

    bench.run("HadamardMultiply", [&]() {
        std::array<float, N> output;
        for (auto i = 0; i < kIterations; ++i)
        {
            sfFDN::HadamardMultiply(kInput, output);
            nanobench::doNotOptimizeAway(output);
        }
    });

    bench.run("WalshHadamardTransform", [&]() {
        std::array<float, N> inout;
        for (auto i = 0; i < N; ++i)
        {
            inout[i] = kInput[i];
        }
        for (auto i = 0; i < kIterations; ++i)
        {
            sfFDN::WalshHadamardTransform(inout);
            nanobench::doNotOptimizeAway(inout);
        }
    });
}