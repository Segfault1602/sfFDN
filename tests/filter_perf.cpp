#include "doctest.h"
#include "nanobench.h"

#include <random>

#include <filterbank.h>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FilterBankPerf")
{
    constexpr size_t N = 16;
    fdn::FilterBank filter_bank(N);

    float pole = 0.90;
    for (size_t i = 0; i < N; i++)
    {
        fdn::OnePoleFilter* filter = new fdn::OnePoleFilter();
        filter->SetCoefficients(1 - pole, -pole);
        filter_bank.SetFilter(i, filter);
        pole -= 0.01;
    }

    constexpr size_t size = 48000;
    std::vector<float> input(N * size, 0);

    // fill input with random values
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    nanobench::Bench bench;
    bench.title("FilterBank perf");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");

    bench.run("FilterBank", [&] {
        std::vector<float> output(N * size, 0);
        filter_bank.Clear();
        filter_bank.Tick(input, output);
        nanobench::doNotOptimizeAway(output);
    });
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

    fdn::CascadedBiquads filter_bank;
    std::vector<float> coeffs;
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs.push_back(sos[i][0] / sos[i][3]);
        coeffs.push_back(sos[i][1] / sos[i][3]);
        coeffs.push_back(sos[i][2] / sos[i][3]);
        coeffs.push_back(sos[i][4] / sos[i][3]);
        coeffs.push_back(sos[i][5] / sos[i][3]);
    }

    filter_bank.SetCoefficients(sos.size(), coeffs);

    constexpr size_t SR = 48000;
    constexpr size_t block_size = 512;
    constexpr size_t ITER = ((SR / block_size) + 1) * block_size; // 1 second at 48kHz
    std::vector<float> input(ITER, 0);

    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(ITER, 0);

    nanobench::Bench bench;
    bench.title("CascadedBiquads perf");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    bench.relative(true);

    bench.run("CascadedBiquads - Samples", [&] {
        filter_bank.Clear();

        for (size_t i = 0; i < ITER; i++)
        {
            output[i] = filter_bank.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(output);
    });

    bench.run("CascadedBiquads - 512 Block", [&] {
        filter_bank.Clear();

        for (size_t i = 0; i < input.size(); i += block_size)
        {
            std::span<float> input_span{input.data() + i, block_size};
            std::span<float> output_span{output.data() + i, block_size};
            filter_bank.ProcessBlock(input_span.data(), output_span.data(), block_size);
        }

        nanobench::doNotOptimizeAway(output);
    });
}