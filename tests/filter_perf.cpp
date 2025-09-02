#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <random>

#include "rng.h"
#include "sffdn/sffdn.h"

#include "filter_coeffs.h"
#include "test_utils.h"

#include <Accelerate/accelerate.h>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FilterBankPerf")
{
    constexpr uint32_t N = 16;

    auto filter_bank = GetFilterBank(N, 11);

    constexpr uint32_t kSampleToProcess = 32768;

    nanobench::Bench bench;
    bench.title("FilterBank perf");
    bench.minEpochIterations(100);
    bench.relative(true);
    bench.timeUnit(1us, "us");
    bench.batch(kSampleToProcess);

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256};
    sfFDN::RNG rng;

    for (const auto& block_size : kBlockSizes)
    {
        std::vector<float> input(block_size * N, 0);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = rng.NextFloat();
        }
        std::vector<float> output(block_size * N, 0);

        bench.run("FilterBank - Block Size " + std::to_string(block_size), [&] {
            const uint32_t num_blocks = kSampleToProcess / block_size;
            assert(kSampleToProcess % block_size == 0);

            for (auto i = 0; i < num_blocks; ++i)
            {
                sfFDN::AudioBuffer input_buffer(block_size, N, input);
                sfFDN::AudioBuffer output_buffer(block_size, N, output);
                filter_bank->Process(input_buffer, output_buffer);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("CascadedBiquadsPerf")
{
    // clang-format off
    constexpr std::array<std::array<float, 6>,11> sos = {{
        {0.81751023887136, 0.f,             0.f,             1.f,              0.f,             0.f},
        {1.03123539966583, -2.05357246743096, 1.022375294192310, 1.03111929845434, -2.05357345199080, 1.02249041084395},
        {1.01622872208192, -2.02365307479989, 1.007493166706850, 1.01612692482198, -2.02365307479989, 1.00759496396680},
        {1.02974305306051, -2.04156824876738, 1.012098520888300, 1.02938518464746, -2.04156824876738, 1.01245638930135},
        {1.03938843409774, -2.04233625493554, 1.004041899029330, 1.03864517487749, -2.04233625493554, 1.00478515824958},
        {1.05902204811827, -2.04269511977105, 0.988056022939481, 1.05740876007274, -2.04269511977105, 0.989669310985015},
        {1.07201865801626, -1.99022403375181, 0.935378940468472, 1.07151604544293, -1.99022403375181, 0.935881553041804},
        {1.12290898014521, -1.91155847686232, 0.856081978411337, 1.12575666122989, -1.91155847686232, 0.853234297326652},
        {1.20682751196864, -1.65249906638422, 0.701314049656436, 1.23174882339560, -1.65249906638422, 0.676392738229472},
        {1.43968619970461, -0.92491012494636, 0.410134050188126, 1.52666454179014, -0.924910124946368 ,0.323155708102591},
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375}
    }};
    // clang-format on

    sfFDN::CascadedBiquads filter_bank;
    std::vector<float> coeffs;
    for (const auto& biquad : sos)
    {
        coeffs.push_back(biquad[0] / biquad[3]);
        coeffs.push_back(biquad[1] / biquad[3]);
        coeffs.push_back(biquad[2] / biquad[3]);
        coeffs.push_back(biquad[4] / biquad[3]);
        coeffs.push_back(biquad[5] / biquad[3]);
    }

    filter_bank.SetCoefficients(sos.size(), coeffs);

    constexpr uint32_t SR = 48000;
    constexpr uint32_t kBlockSize = 128;
    std::vector<float> input(kBlockSize, 0);

    // Fill with white noise
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> dist(0, 0.1);
    for (auto i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(kBlockSize, 0);

    nanobench::Bench bench;
    bench.title("CascadedBiquads perf");
    bench.batch(kBlockSize);
    bench.minEpochIterations(200000);

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    bench.run("CascadedBiquads", [&] { filter_bank.Process(input_buffer, output_buffer); });
}

#if 1
TEST_CASE("VDSP_FilterBank")
{
    constexpr uint32_t N = 6;  // number of channels
    constexpr uint32_t M = 11; // number of section

    std::vector<float> delays((2 * M) + 2, 0.f);
    std::vector<double> coeffs;
    coeffs.reserve(N * M * 5);

    for (auto i = 0; i < N; ++i)
    {
        auto sos = k_h001_AbsorbtionSOS[i];
        REQUIRE(sos.size() == M);

        for (auto j = 0; j < M; ++j)
        {
            coeffs.push_back(sos[j][0] / sos[j][3]);
            coeffs.push_back(sos[j][1] / sos[j][3]);
            coeffs.push_back(sos[j][2] / sos[j][3]);
            coeffs.push_back(sos[j][4] / sos[j][3]);
            coeffs.push_back(sos[j][5] / sos[j][3]);
        }
    }

    vDSP_biquadm_Setup biquad_setup = vDSP_biquadm_CreateSetup(coeffs.data(), M, N);
    REQUIRE(biquad_setup != nullptr);

    constexpr uint32_t kSampleToProcess = 32768;

    nanobench::Bench bench;
    bench.title("FilterBank perf");
    bench.minEpochIterations(100);
    bench.relative(true);
    bench.timeUnit(1us, "us");
    bench.batch(kSampleToProcess);

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256};

    sfFDN::RNG rng;
    for (const auto& block_size : kBlockSizes)
    {
        std::vector<float> input(block_size * N, 0);
        for (auto i = 0; i < input.size(); ++i)
        {
            input[i] = rng.NextFloat();
        }
        std::vector<float> output(block_size * N, 0);

        std::array<const float*, N> input_ptrs;
        std::array<float*, N> output_ptrs;

        for (auto i = 0; i < N; ++i)
        {
            input_ptrs[i] = input.data() + (i * block_size);
            output_ptrs[i] = output.data() + (i * block_size);
        }

        bench.run("FilterBank - Block Size " + std::to_string(block_size), [&] {
            const uint32_t num_blocks = kSampleToProcess / block_size;
            assert(kSampleToProcess % block_size == 0);

            for (auto i = 0; i < num_blocks; ++i)
            {

                vDSP_biquadm(biquad_setup, input_ptrs.data(), 1, output_ptrs.data(), 1, block_size);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }
}
#endif