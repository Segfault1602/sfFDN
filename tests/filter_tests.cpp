#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <limits>

#include <sndfile.h>

#include "sffdn/audio_buffer.h"
#include "sffdn/sffdn.h"

#include "rng.h"

namespace
{
constexpr std::array<std::array<float, 6>, 11> kTestSOS = {
    {{0.81751023887136, 0.f, 0.f, 1.f, 0.f, 0.f},
     {1.03123539966583, -2.05357246743096, 1.022375294192310, 1.03111929845434, -2.05357345199080, 1.02249041084395},
     {1.01622872208192, -2.02365307479989, 1.007493166706850, 1.01612692482198, -2.02365307479989, 1.00759496396680},
     {1.02974305306051, -2.04156824876738, 1.012098520888300, 1.02938518464746, -2.04156824876738, 1.01245638930135},
     {1.03938843409774, -2.04233625493554, 1.004041899029330, 1.03864517487749, -2.04233625493554, 1.00478515824958},
     {1.05902204811827, -2.04269511977105, 0.988056022939481, 1.05740876007274, -2.04269511977105, 0.989669310985015},
     {1.07201865801626, -1.99022403375181, 0.935378940468472, 1.07151604544293, -1.99022403375181, 0.935881553041804},
     {1.12290898014521, -1.91155847686232, 0.856081978411337, 1.12575666122989, -1.91155847686232, 0.853234297326652},
     {1.20682751196864, -1.65249906638422, 0.701314049656436, 1.23174882339560, -1.65249906638422, 0.676392738229472},
     {1.43968619970461, -0.92491012494636, 0.410134050188126, 1.52666454179014, -0.924910124946368, 0.323155708102591},
     {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284,
      0.475604303744375}}};

constexpr std::array<float, 32> kTestSOSExpectedOutput = {
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

    constexpr uint32_t kSize = 8;
    std::array<float, kSize> input = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::array<float, kSize> output{};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);

    filter.Process(input_buffer, output_buffer);

    constexpr std::array<float, kSize> kExpectedOutput = {0.1000, 0.0900, 0.0810, 0.0729,
                                                          0.0656, 0.0590, 0.0531, 0.0478};

    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(kExpectedOutput[i], 0.0001));
    }
}

TEST_CASE("FirFilter")
{
    constexpr uint32_t kFirSize = 64;
    sfFDN::Fir filter;
    std::vector<float> ir(kFirSize, 0.f);

    sfFDN::RNG rng;
    for (auto& coeff : ir)
    {
        coeff = rng();
    }
    filter.SetCoefficients(ir);

    constexpr uint32_t kSize = 128;
    std::array<float, kSize> input = {0.f};
    input[0] = 1.f;
    std::array<float, kSize> output{};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);

    filter.Process(input_buffer, output_buffer);

    for (auto i = 0u; i < kFirSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(ir[i], 1e-5));
    }

    for (auto i = kFirSize; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(0.f, 1e-5));
    }
}

TEST_CASE("SparseFirFilter")
{
    constexpr uint32_t kFirSize = 64;
    std::vector<float> ir(kFirSize, 0.f);
    std::vector<float> sparse_ir;
    std::vector<uint32_t> indices;

    sfFDN::RNG rng;
    for (auto i = 0u; i < kFirSize; i++)
    {
        if (i % 4 == 0)
        {
            auto s = rng();
            ir[i] = s;
            sparse_ir.push_back(s);
            indices.push_back(i);
        }
    }
    sfFDN::Fir filter;
    filter.SetCoefficients(ir);

    sfFDN::SparseFir sparse_filter;
    sparse_filter.SetCoefficients(sparse_ir, indices);

    constexpr uint32_t kSize = 128;
    std::array<float, kSize> input = {0.f};
    input[0] = 1.f;
    std::array<float, kSize> output{};
    std::array<float, kSize> sparse_output{};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);
    sfFDN::AudioBuffer sparse_output_buffer(kSize, 1, sparse_output);

    filter.Process(input_buffer, output_buffer);
    sparse_filter.Process(input_buffer, sparse_output_buffer);

    for (auto i = 0u; i < kFirSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(ir[i], 1e-5));
        REQUIRE_THAT(sparse_output[i], Catch::Matchers::WithinAbs(ir[i], 1e-5));
    }

    for (auto i = kFirSize; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(0.f, 1e-5));
        REQUIRE_THAT(sparse_output[i], Catch::Matchers::WithinAbs(0.f, 1e-5));
    }
}

TEST_CASE("SchroederAllpass")
{
    sfFDN::SchroederAllpass filter(5, -0.9);

    constexpr uint32_t kSize = 18;
    std::array<float, kSize> input = {0.f};
    input[0] = 1.f;
    constexpr std::array<float, kSize> kExpectedOutput = {0.9, 0,      0, 0, 0, 0.19, 0,      0, 0,
                                                          0,   -0.171, 0, 0, 0, 0,    0.1539, 0, 0};

    for (auto i = 0u; i < kSize; ++i)
    {
        float out = filter.Tick(input[i]);
        REQUIRE_THAT(out, Catch::Matchers::WithinAbs(kExpectedOutput[i], 0.0001));
    }

    sfFDN::SchroederAllpass filter_block(5, -0.9);
    std::array<float, kSize> output{};
    filter_block.ProcessBlock(input, output);

    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(kExpectedOutput[i], 0.0001));
    }
}

TEST_CASE("SchroederAllpassSection")
{
    sfFDN::SchroederAllpassSection filter(2);

    constexpr std::array<uint32_t, 2> kDelays = {3, 5};
    constexpr std::array<float, 2> kGains = {0.9f, 0.8f};

    filter.SetDelays(kDelays);
    filter.SetGains(kGains);

    constexpr uint32_t kSize = 12;
    std::array<float, kSize> input = {0.f};
    std::array<float, kSize> output = {0.f};
    input[0] = 1.f;
    constexpr std::array<float, kSize> kExpectedOutput = {0.72,    0, 0,      -0.152,   0,       -0.324,
                                                          -0.1368, 0, 0.0684, -0.12312, -0.2592, 0.06156};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);

    filter.Process(input_buffer, output_buffer);

    for (auto i = 0u; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(kExpectedOutput[i], 0.0001));
    }
}

TEST_CASE("ParallelSchroederAllpassSection")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 8;

    sfFDN::ParallelSchroederAllpassSection filter(kChannelCount, 1);
    std::array<uint32_t, kChannelCount> delays = {2, 3, 4, 5};
    std::array<float, kChannelCount> gains = {-0.9, -0.8, -0.7, -0.6};
    filter.SetDelays(delays);
    filter.SetGains(gains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    filter.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> kOut0Expected = {0.9, 0, 0.19, 0, -0.171, 0, 0.1539, 0};
    constexpr std::array<float, kBlockSize> kOut1Expected = {0.8, 0, 0, 0.36, 0, 0, -0.288, 0};
    constexpr std::array<float, kBlockSize> kOut2Expected = {0.7, 0, 0, 0, 0.51, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kOut3Expected = {0.6, 0, 0, 0, 0, 0.64, 0, 0};

    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[0 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(kOut0Expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[1 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(kOut1Expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[2 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(kOut2Expected[j], std::numeric_limits<float>::epsilon()));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[3 * kBlockSize + j],
                     Catch::Matchers::WithinAbs(kOut3Expected[j], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("ParallelSchroederAllpassSection_Order2")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 8;

    sfFDN::ParallelSchroederAllpassSection filter(kChannelCount, 2);
    std::array<uint32_t, kChannelCount * 2> delays = {2, 5, 4, 1, 4, 6, 2, 5};
    std::array<float, kChannelCount> gains = {0.9, 0.8, 0.7, 0.6};
    filter.SetDelays(delays);
    filter.SetGains(gains);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    filter.Process(input_buffer, output_buffer);

    constexpr std::array<float, kBlockSize> kOut0Expected = {0.810000,  0.000000,  -0.171000, 0.000000,
                                                             -0.153900, -0.171000, -0.138510, 0.036100};
    constexpr std::array<float, kBlockSize> kOut1Expected = {0.640000,  -0.288000, -0.230400, -0.184320,
                                                             -0.435456, 0.011635,  0.009308,  0.007447};
    constexpr std::array<float, kBlockSize> kOut2Expected = {0.490000,  0.000000, 0.000000,  0.000000,
                                                             -0.357000, 0.000000, -0.357000, 0.000000};
    constexpr std::array<float, kBlockSize> kOut3Expected = {0.360000,  0.000000,  -0.384000, 0.000000,
                                                             -0.230400, -0.384000, -0.138240, 0.409600};

    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[0 * kBlockSize + j], Catch::Matchers::WithinAbs(kOut0Expected[j], 1e-5f));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[1 * kBlockSize + j], Catch::Matchers::WithinAbs(kOut1Expected[j], 1e-5f));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[2 * kBlockSize + j], Catch::Matchers::WithinAbs(kOut2Expected[j], 1e-5f));
    }
    for (auto j = 0u; j < kBlockSize; ++j)
    {
        REQUIRE_THAT(output[3 * kBlockSize + j], Catch::Matchers::WithinAbs(kOut3Expected[j], 1e-5f));
    }
}

TEST_CASE("FilterBank")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kBlockSize = 8;
    sfFDN::FilterBank filter_bank;

    float pole = 0.9;
    for (auto i = 0u; i < kChannelCount; i++)
    {
        auto filter = std::make_unique<sfFDN::OnePoleFilter>();
        filter->SetCoefficients(1 - pole, -pole);
        filter_bank.AddFilter(std::move(filter));
        pole -= 0.1;
    }

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    // Input vector is deinterleaved by delay line: {d0_0, d0_1, d0_2, ..., d1_0, d1_1, d1_2, ..., dN_0, dN_1, dN_2}
    for (auto i = 0u; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    filter_bank.Process(input_buffer, output_buffer);

    constexpr std::array<float, 32> kExpectedOutput = {
        0.1,    0.09,   0.081,  0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.2000, 0.1600, 0.1280,
        0.1024, 0.0819, 0.0655, 0.0524, 0.0419,  0.3000,   0.2100,    0.1470,     0.1029, 0.0720, 0.0504,
        0.0353, 0.0247, 0.4000, 0.2400, 0.1440,  0.0864,   0.0518,    0.0311,     0.0187, 0.0112};

    for (auto i = 0u; i < kExpectedOutput.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(kExpectedOutput[i], 0.0001));
    }
}

TEST_CASE("CascadedBiquads")
{
    sfFDN::CascadedBiquads filter;

    std::vector<float> coeffs;
    for (const auto& sos : kTestSOS)
    {
        coeffs.push_back(sos[0] / sos[3]);
        coeffs.push_back(sos[1] / sos[3]);
        coeffs.push_back(sos[2] / sos[3]);
        coeffs.push_back(sos[4] / sos[3]);
        coeffs.push_back(sos[5] / sos[3]);
    }

    filter.SetCoefficients(kTestSOS.size(), coeffs);

    constexpr uint32_t kSize = 32;
    std::array<float, kSize> input = {0};
    input[0] = 1.f;
    std::array<float, kSize> output{};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);

    filter.Process(input_buffer, output_buffer);

    REQUIRE(kSize == kTestSOSExpectedOutput.size());
    for (auto i = 0u; i < kTestSOSExpectedOutput.size(); ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(kTestSOSExpectedOutput[i], 0.0001));
    }
}

TEST_CASE("IIRFilterBank")
{
    constexpr uint32_t kChannelCount = 6;
    constexpr uint32_t kStageCount = kTestSOS.size();
    std::vector<float> coeffs;
    for (auto n = 0; n < kChannelCount; ++n)
    {
        for (auto i = 0u; i < kStageCount; i++)
        {
            coeffs.push_back(kTestSOS[i][0] / kTestSOS[i][3]);
            coeffs.push_back(kTestSOS[i][1] / kTestSOS[i][3]);
            coeffs.push_back(kTestSOS[i][2] / kTestSOS[i][3]);
            coeffs.push_back(kTestSOS[i][4] / kTestSOS[i][3]);
            coeffs.push_back(kTestSOS[i][5] / kTestSOS[i][3]);
        }
    }

    sfFDN::IIRFilterBank filter_bank;
    filter_bank.SetFilter(coeffs, kChannelCount, kStageCount);

    constexpr uint32_t kBlockSize = 16;
    std::vector<float> input(kBlockSize * kChannelCount, 0.f);
    std::vector<float> output(kBlockSize * kChannelCount, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    for (auto i = 0u; i < kChannelCount; ++i)
    {
        input_buffer.GetChannelSpan(i)[0] = 1.f;
    }

    filter_bank.Process(input_buffer, output_buffer);

    for (auto i = 0u; i < kBlockSize; ++i)
    {
        for (auto n = 0; n < kChannelCount; ++n)
        {
            REQUIRE_THAT(output_buffer.GetChannelSpan(n)[i],
                         Catch::Matchers::WithinAbs(kTestSOSExpectedOutput[i], 0.0001));
        }
    }
}
