#include "doctest.h"
#include "nanobench.h"

#include <random>

#include <Accelerate/Accelerate.h>

#include "CMSIS/filtering_functions.h"
#include <filterbank.h>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FilterBankPerf" * doctest::skip(true))
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

TEST_CASE("vDSPBiquad")
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
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375},
    }};
    // clang-format on
    constexpr size_t num_stage = 11;
    double coeffs[5 * num_stage] = {0};
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs[i * 5 + 0] = sos[i][0] / sos[i][3];
        coeffs[i * 5 + 1] = sos[i][1] / sos[i][3];
        coeffs[i * 5 + 2] = sos[i][2] / sos[i][3];
        coeffs[i * 5 + 3] = sos[i][4] / sos[i][3];
        coeffs[i * 5 + 4] = sos[i][5] / sos[i][3];
    }

    auto biquad_setup = vDSP_biquad_CreateSetup(coeffs, num_stage);

    constexpr size_t size = 48000;
    std::vector<float> input(size, 0);

    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(size, 0);

    nanobench::Bench bench;
    bench.title("vDSP Perf");
    bench.minEpochIterations(1000);
    bench.timeUnit(1ms, "ms");
    bench.run("vDSP", [&] {
        float delays[2 * num_stage + 2] = {0};
        vDSP_biquad(biquad_setup, delays, input.data(), 1, output.data(), 1, size);
    });

    vDSP_biquad_DestroySetup(biquad_setup);
}

TEST_CASE("CascadedBiquadVSCmsis" * doctest::skip(true))
{
    constexpr size_t num_stage = 12;
    // clang-format off
    constexpr std::array<std::array<float, 6>,num_stage> sos = {{
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
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375},
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375}
    }};

    float coeffs[5*num_stage] = {0};
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs[i * 5 + 0] = sos[i][0] / sos[i][3];
        coeffs[i * 5 + 1] = sos[i][1] / sos[i][3];
        coeffs[i * 5 + 2] = sos[i][2] / sos[i][3];
        coeffs[i * 5 + 3] = -sos[i][4] / sos[i][3];
        coeffs[i * 5 + 4] = -sos[i][5] / sos[i][3];
    }

    float state[8 * num_stage] = {0};

    float computed_coeffs[8 * num_stage] = {0};
    arm_biquad_cascade_df2T_instance_f32 biquad_instance;
    arm_biquad_cascade_df2T_compute_coefs_f32(num_stage, coeffs, computed_coeffs);

    constexpr size_t size = 48000;
    constexpr size_t block_size = 512;
    std::vector<float> input(size, 0);

    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(size, 0);
    nanobench::Bench bench;
    bench.title("CascadedBiquads vs CMSIS perf");
    bench.minEpochIterations(1000);
    bench.timeUnit(1ms, "ms");
    bench.relative(true);

    bench.run("CMSIS", [&] {
        arm_biquad_cascade_df2T_init_f32(&biquad_instance, num_stage, computed_coeffs, state);
        for (size_t i = 0; i < size; i += block_size)
        {
            arm_biquad_cascade_df2T_f32(&biquad_instance, input.data() + i, output.data() + i, block_size);
        }
        // nanobench::doNotOptimizeAway(output);
    });

    fdn::CascadedBiquads cascaded_biquad;
    std::vector<float> coeffs_vec;
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs_vec.push_back(sos[i][0] / sos[i][3]);
        coeffs_vec.push_back(sos[i][1] / sos[i][3]);
        coeffs_vec.push_back(sos[i][2] / sos[i][3]);
        coeffs_vec.push_back(sos[i][4] / sos[i][3]);
        coeffs_vec.push_back(sos[i][5] / sos[i][3]);
    }

    cascaded_biquad.SetCoefficients(sos.size(), coeffs_vec);

    bench.run("CascadedBiquads", [&] {
        cascaded_biquad.Clear();

        for (size_t i = 0; i < size; i += block_size)
        {
            cascaded_biquad.ProcessBlock(input.data() + i, output.data()+i, block_size);
        }
    });

    // vDSP
    double coeffs_d[5 * num_stage] = {0};
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs_d[i * 5 + 0] = sos[i][0] / sos[i][3];
        coeffs_d[i * 5 + 1] = sos[i][1] / sos[i][3];
        coeffs_d[i * 5 + 2] = sos[i][2] / sos[i][3];
        coeffs_d[i * 5 + 3] = sos[i][4] / sos[i][3];
        coeffs_d[i * 5 + 4] = sos[i][5] / sos[i][3];
    }

    auto biquad_setup = vDSP_biquad_CreateSetup(coeffs_d, num_stage);
    output.clear();
    output.resize(size, 0);

    bench.run("vDSP", [&] {
        float delays[2 * num_stage + 2] = {0};
        for (size_t i = 0; i < size; i += block_size)
        {
            vDSP_biquad(biquad_setup, delays, input.data() + i, 1, output.data() + i, 1, block_size);
        }
    });

    vDSP_biquad_DestroySetup(biquad_setup);
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

    constexpr size_t size = 48000;
    std::vector<float> input(size, 0);

    // Fill with white noise
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.1);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = dist(generator);
    }

    std::vector<float> output(size, 0);

    nanobench::Bench bench;
    bench.title("CascadedBiquads perf");
    bench.minEpochIterations(100);
    bench.timeUnit(1ms, "ms");
    bench.relative(true);

    bench.run("CascadedBiquads - Samples", [&] {
        filter_bank.Clear();

        for (size_t i = 0; i < size; i++)
        {
            output[i] = filter_bank.Tick(input[i]);
        }

        nanobench::doNotOptimizeAway(output);
    });

    constexpr size_t block_size = 512;
    bench.run("CascadedBiquads - 512 Block", [&] {
        filter_bank.Clear();

        size_t block_count = size / block_size;
        for (size_t i = 0; i < block_count; i++)
        {
            filter_bank.ProcessBlock(input.data(), output.data(), block_size);
        }

        // remainder
        size_t remainder = size % block_size;
        if (remainder > 0)
        {
            filter_bank.ProcessBlock(input.data(), output.data(), remainder);
        }

        nanobench::doNotOptimizeAway(output);
    });
}