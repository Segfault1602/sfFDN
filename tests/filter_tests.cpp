#include "doctest.h"

#include <iomanip>
#include <iostream>
#include <random>

#include "audio_buffer.h"
#include "filter.h"
#include "filterbank.h"
#include "schroeder_allpass.h"

#include <sndfile.h>

namespace
{
// clang-format off
    constexpr std::array<std::array<float, 6>,11> kTestSOS = {{
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

const std::vector<float> kTestSOSExpectedOutput = {
    0.678000939417768,     0.0398721002729839,   0.0388255041778860,   0.0242086305009620,   0.0215610414280036,
    0.0164821225299678,    0.0115111695707740,   0.00912522376126048,  0.00764219320916558,  0.00585150622757179,
    0.00406548919279410,   0.00280330418856257,  0.00214252048661309,  0.00188750524502253,  0.00182319004433901,
    0.00180387800104089,   0.00175126815522666,  0.00163622788868539,  0.00146192288654082,  0.00124863755091232,
    0.00102159827055317,   0.000803109246775104, 0.000608757012952238, 0.000446606715608455, 0.000318244039595866,
    0.000220687308459613,  0.000148431854954261, 9.51663729241437e-05, 5.49631625671496e-05, 2.29367625168784e-05,
    -4.52654209677817e-06, -2.98274633506682e-05};

} // namespace

TEST_CASE("OnePoleFilter")
{
    sfFDN::OnePoleFilter filter;
    filter.SetCoefficients(0.1, -0.9);

    constexpr size_t size = 8;
    constexpr std::array<float, size> input = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::array<float, size> output;

    filter.ProcessBlock(input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {0.1000, 0.0900, 0.0810, 0.0729,
                                                         0.0656, 0.0590, 0.0531, 0.0478};

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("SchroederAllpass")
{
    sfFDN::SchroederAllpass filter(5, 0.9);

    constexpr size_t size = 18;
    std::array<float, size> input = {0.f};
    input[0] = 1.f;
    constexpr std::array<float, size> expected_output = {0.9, 0,      0, 0, 0, 0.19, 0,      0, 0,
                                                         0,   -0.171, 0, 0, 0, 0,    0.1539, 0, 0};

    for (size_t i = 0; i < size; ++i)
    {
        float out = filter.Tick(input[i]);
        CHECK(out == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }

    sfFDN::SchroederAllpass filter_block(5, 0.9);
    std::array<float, size> output;
    filter_block.ProcessBlock(input, output);

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("SchroederAllpassSection")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;

    sfFDN::SchroederAllpassSection filter(N);
    std::array<size_t, N> delays = {2, 3, 4, 5};
    std::array<float, N> gains = {0.9, 0.8, 0.7, 0.6};
    filter.SetDelays(delays);
    filter.SetGains(gains);

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    filter.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> out0_expected = {0.9, 0, 0.19, 0, -0.171, 0, 0.1539, 0};
    constexpr std::array<float, kBlockSize> out1_expected = {0.8, 0, 0, 0.36, 0, 0, -0.288, 0};
    constexpr std::array<float, kBlockSize> out2_expected = {0.7, 0, 0, 0, 0.51, 0, 0, 0};
    constexpr std::array<float, kBlockSize> out3_expected = {0.6, 0, 0, 0, 0, 0.64, 0, 0};

    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[0 * kBlockSize + j] == doctest::Approx(out0_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[1 * kBlockSize + j] == doctest::Approx(out1_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[2 * kBlockSize + j] == doctest::Approx(out2_expected[j]).epsilon(0.01));
    }
    for (size_t j = 0; j < kBlockSize; ++j)
    {
        CHECK(output[3 * kBlockSize + j] == doctest::Approx(out3_expected[j]).epsilon(0.01));
    }
}

TEST_CASE("FilterBank")
{
    constexpr size_t N = 4;
    constexpr size_t kBlockSize = 8;
    sfFDN::FilterBank filter_bank;

    float pole = 0.9;
    for (size_t i = 0; i < N; i++)
    {
        auto filter = std::make_unique<sfFDN::OnePoleFilter>();
        filter->SetCoefficients(1 - pole, -pole);
        filter_bank.AddFilter(std::move(filter));
        pole -= 0.1;
    }

    std::vector<float> input(N * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (size_t i = 0; i < N; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(N * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, N, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, N, output);

    filter_bank.Process(input_buffer, output_buffer);

    std::vector<float> expected_output = {0.1,    0.09,   0.081,  0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969,
                                          0.2000, 0.1600, 0.1280, 0.1024, 0.0819,  0.0655,   0.0524,    0.0419,
                                          0.3000, 0.2100, 0.1470, 0.1029, 0.0720,  0.0504,   0.0353,    0.0247,
                                          0.4000, 0.2400, 0.1440, 0.0864, 0.0518,  0.0311,   0.0187,    0.0112};

    for (size_t i = 0; i < expected_output.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("CascadedBiquads")
{
    sfFDN::CascadedBiquads filter;

    std::vector<float> coeffs;
    for (size_t i = 0; i < kTestSOS.size(); i++)
    {
        coeffs.push_back(kTestSOS[i][0] / kTestSOS[i][3]);
        coeffs.push_back(kTestSOS[i][1] / kTestSOS[i][3]);
        coeffs.push_back(kTestSOS[i][2] / kTestSOS[i][3]);
        coeffs.push_back(kTestSOS[i][4] / kTestSOS[i][3]);
        coeffs.push_back(kTestSOS[i][5] / kTestSOS[i][3]);
    }

    filter.SetCoefficients(kTestSOS.size(), coeffs);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    filter.ProcessBlock(input.data(), output.data(), size);

    REQUIRE(size == kTestSOSExpectedOutput.size());
    for (size_t i = 0; i < kTestSOSExpectedOutput.size(); ++i)
    {
        CHECK(output[i] == doctest::Approx(kTestSOSExpectedOutput[i]).epsilon(0.0001));
    }
}