#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "rng.h"
#include "sffdn/delay_utils.h"
#include "sffdn/sffdn.h"

#include "filter_coeffs.h"
#include "test_utils.h"

#include <iostream>
#include <random>

#ifdef __APPLE__
#include <Accelerate/accelerate.h>
#endif

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FilterBankPerf")
{
    constexpr uint32_t kChannelCount = 16;

    auto filter_bank = GetFilterBank(kChannelCount, 11);

    constexpr uint32_t kSampleToProcess = 512;

    nanobench::Bench bench;
    bench.title("FilterBank perf - Block Size Comparison");
    bench.minEpochIterations(200);
    bench.relative(true);
    bench.timeUnit(1us, "us");

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256};
    sfFDN::RNG rng;

    for (const auto& block_size : kBlockSizes)
    {
        std::vector<float> input(block_size * kChannelCount, 0);
        for (float& i : input)
        {
            i = rng();
        }
        std::vector<float> output(block_size * kChannelCount, 0);

        bench.run("FilterBank - Block Size " + std::to_string(block_size), [&] {
            const uint32_t num_blocks = kSampleToProcess / block_size;
            assert(kSampleToProcess % block_size == 0);

            for (auto i = 0u; i < num_blocks; ++i)
            {
                sfFDN::AudioBuffer input_buffer(block_size, kChannelCount, input);
                sfFDN::AudioBuffer output_buffer(block_size, kChannelCount, output);
                filter_bank->Process(input_buffer, output_buffer);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }
}

TEST_CASE("IIRFilterBankPerf")
{
    constexpr uint32_t kChannelCount = 16;
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kBlockSize = 128;

    constexpr std::array<float, 10> kRT60s = {2.f, 2.1f, 2.5f, 2.f, 1.5f, 1.f, 0.8f, 0.5f, 0.3f, 0.21f};
    auto delays = sfFDN::GetDelayLengths(kChannelCount, 500, 5000, sfFDN::DelayLengthType::Uniform);

    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (auto i = 0u; i < kChannelCount; i++)
    {
        auto filter_coeffs = sfFDN::GetTwoFilter(kRT60s, delays[i], kSampleRate);
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();

        auto order = filter_coeffs.size() / 6;
        filter->SetCoefficients(order, filter_coeffs);
        filter_bank->AddFilter(std::move(filter));
    }

    nanobench::Bench bench;
    bench.title("FilterBank vs IIRFilterBank perf");
    bench.minEpochIterations(5000);
    bench.relative(true);
    bench.timeUnit(1us, "us");

    std::vector<float> input(kBlockSize * kChannelCount, 0);
    sfFDN::RNG rng;
    for (float& i : input)
    {
        i = rng();
    }
    std::vector<float> output(kBlockSize * kChannelCount, 0);

    bench.run("FilterBank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);
        filter_bank->Process(input_buffer, output_buffer);

        nanobench::doNotOptimizeAway(output);
    });

    auto iir_filter_bank = std::make_unique<sfFDN::IIRFilterBank>();
    std::vector<float> coeffs;
    for (auto i = 0u; i < kChannelCount; i++)
    {
        auto filter_coeffs = sfFDN::GetTwoFilter(kRT60s, delays[i], kSampleRate);
        coeffs.insert(coeffs.end(), filter_coeffs.begin(), filter_coeffs.end());
    }
    iir_filter_bank->SetFilter(coeffs, kChannelCount, 11);

    bench.run("IIRFilterBank", [&] {
        sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);
        iir_filter_bank->Process(input_buffer, output_buffer);

        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("CascadedBiquadsPerf")
{
    // clang-format off
    constexpr std::array<std::array<float, 6>,11> kSOS = {{
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
    for (const auto& biquad : kSOS)
    {
        coeffs.push_back(biquad[0] / biquad[3]);
        coeffs.push_back(biquad[1] / biquad[3]);
        coeffs.push_back(biquad[2] / biquad[3]);
        coeffs.push_back(biquad[4] / biquad[3]);
        coeffs.push_back(biquad[5] / biquad[3]);
    }

    filter_bank.SetCoefficients(kSOS.size(), coeffs);

    constexpr uint32_t kBlockSize = 128;
    std::vector<float> input(kBlockSize, 0);

    // Fill with white noise
    sfFDN::RNG generator;
    for (auto& i : input)
    {
        i = generator();
    }

    std::vector<float> output(kBlockSize, 0);

    nanobench::Bench bench;
    bench.title("CascadedBiquads perf");
    // bench.batch(kBlockSize);
    bench.minEpochIterations(20000);
    bench.timeUnit(1us, "us");

    sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);

    bench.run("CascadedBiquads", [&] { filter_bank.Process(input_buffer, output_buffer); });
}

TEST_CASE("FirFilter")
{
    nanobench::Bench bench;
    bench.title("Fir perf");
    bench.minEpochIterations(50000);
    bench.relative(true);
    bench.timeUnit(1us, "us");

    constexpr std::array kFirSizes = {16, 32, 64, 128, 256};
    sfFDN::RNG rng;

    for (const auto& fir_size : kFirSizes)
    {
        std::vector<float> ir(fir_size, 0.f);
        for (auto& coeff : ir)
        {
            coeff = rng();
        }
        sfFDN::Fir filter;
        filter.SetCoefficients(ir);

        constexpr uint32_t kSize = 128;
        std::array<float, kSize> input = {0.f};
        input[0] = 1.f;
        std::array<float, kSize> output{};

        sfFDN::AudioBuffer input_buffer(kSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kSize, 1, output);

        bench.run("Fir - Size " + std::to_string(fir_size), [&] { filter.Process(input_buffer, output_buffer); });
    }
}

TEST_CASE("FirFilterSparse")
{
    nanobench::Bench bench;
    bench.title("Fir sparse perf");
    bench.minEpochIterations(50000);
    bench.relative(true);
    bench.timeUnit(1us, "us");

    constexpr std::array kFirSizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    constexpr uint32_t kFirTapCount = 32;
    sfFDN::RNG rng;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto& fir_size : kFirSizes)
    {
        std::uniform_int_distribution<> distribution(0, fir_size - 1);
        std::vector<float> ir;
        std::vector<uint32_t> indices;

        for (auto i = 0u; i < kFirTapCount; i++)
        {
            ir.push_back(rng());
            auto idx = distribution(gen);
            indices.insert(std::upper_bound(indices.begin(), indices.end(), idx), idx);
        }

        sfFDN::SparseFir filter;
        filter.SetCoefficients(ir, indices);

        constexpr uint32_t kSize = 128;
        std::array<float, kSize> input = {0.f};
        input[0] = 1.f;
        std::array<float, kSize> output{};

        sfFDN::AudioBuffer input_buffer(kSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kSize, 1, output);

        bench.run("Fir - Size " + std::to_string(fir_size), [&] { filter.Process(input_buffer, output_buffer); });
    }
}

TEST_CASE("ParallelSchroederAllpassSection")
{
    constexpr uint32_t kChannelCount = 16;
    constexpr uint32_t kBlockSize = 128;

    sfFDN::ParallelSchroederAllpassSection filter(kChannelCount, 1);
    std::vector<uint32_t> delays =
        sfFDN::GetDelayLengths(kChannelCount, kBlockSize, 1000, sfFDN::DelayLengthType::Uniform);
    std::array<float, kChannelCount> gains{};
    gains.fill(0.7f);

    filter.SetDelays(delays);
    filter.SetGains(gains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    nanobench::Bench bench;
    bench.title("ParallelSchroederAllpassSection perf");
    bench.minEpochIterations(5000);
    bench.timeUnit(1us, "us");

    bench.run("ParallelSchroederAllpassSection", [&] { filter.Process(input_buffer, output_buffer); });
}

#ifdef __APPLE__
TEST_CASE("VDSP_FilterBank")
{
    constexpr uint32_t N = 16; // number of channels
    constexpr uint32_t M = 11; // number of section

    std::vector<float> delays((2 * M) + 2, 0.f);
    std::vector<double> coeffs;
    coeffs.reserve(N * M * 5);

    for (auto i = 0u; i < N; ++i)
    {
        auto sos = k_h001_AbsorbtionSOS[i];
        REQUIRE(sos.size() == M);

        for (auto j = 0u; j < M; ++j)
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

    constexpr uint32_t kSampleToProcess = 512;

    nanobench::Bench bench;
    bench.title("vDSP FilterBank perf");
    bench.minEpochIterations(1000);
    bench.relative(true);
    bench.timeUnit(1us, "us");

    constexpr std::array kBlockSizes = {1, 4, 8, 16, 32, 64, 128, 256};

    sfFDN::RNG rng;
    for (const auto& block_size : kBlockSizes)
    {
        std::vector<float> input(block_size * N, 0);
        for (float& i : input)
        {
            i = rng();
        }
        std::vector<float> output(block_size * N, 0);

        std::array<const float*, N> input_ptrs{};
        std::array<float*, N> output_ptrs{};

        for (auto i = 0u; i < N; ++i)
        {
            input_ptrs[i] = std::span(input).subspan(i * block_size, block_size).data();
            output_ptrs[i] = std::span(output).subspan(i * block_size, block_size).data();
        }

        bench.run("FilterBank - Block Size " + std::to_string(block_size), [&] {
            const uint32_t num_blocks = kSampleToProcess / block_size;
            assert(kSampleToProcess % block_size == 0);

            for (auto i = 0u; i < num_blocks; ++i)
            {
                vDSP_biquadm(biquad_setup, input_ptrs.data(), 1, output_ptrs.data(), 1, block_size);
            }
            nanobench::doNotOptimizeAway(output);
        });
    }
}
#endif