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
    {{0.81751023887136f, 0.f, 0.f, 1.f, 0.f, 0.f},
     {1.03123539966583f, -2.05357246743096f, 1.022375294192310f, 1.03111929845434f, -2.05357345199080f,
      1.02249041084395f},
     {1.01622872208192f, -2.02365307479989f, 1.007493166706850f, 1.01612692482198f, -2.02365307479989f,
      1.00759496396680f},
     {1.02974305306051f, -2.04156824876738f, 1.012098520888300f, 1.02938518464746f, -2.04156824876738f,
      1.01245638930135f},
     {1.03938843409774f, -2.04233625493554f, 1.004041899029330f, 1.03864517487749f, -2.04233625493554f,
      1.00478515824958f},
     {1.05902204811827f, -2.04269511977105f, 0.988056022939481f, 1.05740876007274f, -2.04269511977105f,
      0.989669310985015f},
     {1.07201865801626f, -1.99022403375181f, 0.935378940468472f, 1.07151604544293f, -1.99022403375181f,
      0.935881553041804f},
     {1.12290898014521f, -1.91155847686232f, 0.856081978411337f, 1.12575666122989f, -1.91155847686232f,
      0.853234297326652f},
     {1.20682751196864f, -1.65249906638422f, 0.701314049656436f, 1.23174882339560f, -1.65249906638422f,
      0.676392738229472f},
     {1.43968619970461f, -0.92491012494636f, 0.410134050188126f, 1.52666454179014f, -0.924910124946368f,
      0.323155708102591f},
     {2.42350220912989f, -0.09096516658686f, 0.416410844594722f, 2.70192581010466f, -0.428582226711284f,
      0.475604303744375f}}};

constexpr std::array<float, 32> kTestSOSExpectedOutput = {
    0.678000939417768f,     0.0398721002729839f,   0.0388255041778860f,   0.0242086305009620f,   0.0215610414280036f,
    0.0164821225299678f,    0.0115111695707740f,   0.00912522376126048f,  0.00764219320916558f,  0.00585150622757179f,
    0.00406548919279410f,   0.00280330418856257f,  0.00214252048661309f,  0.00188750524502253f,  0.00182319004433901f,
    0.00180387800104089f,   0.00175126815522666f,  0.00163622788868539f,  0.00146192288654082f,  0.00124863755091232f,
    0.00102159827055317f,   0.000803109246775104f, 0.000608757012952238f, 0.000446606715608455f, 0.000318244039595866f,
    0.000220687308459613f,  0.000148431854954261f, 9.51663729241437e-05f, 5.49631625671496e-05f, 2.29367625168784e-05f,
    -4.52654209677817e-06f, -2.98274633506682e-05f};

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

    constexpr std::array<float, kSize> kExpectedOutput = {0.1000f, 0.0900f, 0.0810f, 0.0729f,
                                                          0.0656f, 0.0590f, 0.0531f, 0.0478f};

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
    constexpr std::array<float, kSize> kExpectedOutput = {0.9f, 0.f,     0.f, 0.f, 0.f, 0.19f, 0.f,     0.f, 0.f,
                                                          0.f,  -0.171f, 0.f, 0.f, 0.f, 0.f,   0.1539f, 0.f, 0.f};

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
    constexpr std::array<float, kSize> kExpectedOutput = {0.72f,    0.f, 0.f,     -0.152f,   0.f,      -0.324f,
                                                          -0.1368f, 0.f, 0.0684f, -0.12312f, -0.2592f, 0.06156f};

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
    std::array<float, kChannelCount> gains = {-0.9f, -0.8f, -0.7f, -0.6f};
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

    constexpr std::array<float, kBlockSize> kOut0Expected = {0.9f, 0, 0.19f, 0, -0.171f, 0, 0.1539f, 0};
    constexpr std::array<float, kBlockSize> kOut1Expected = {0.8f, 0, 0, 0.36f, 0, 0, -0.288f, 0};
    constexpr std::array<float, kBlockSize> kOut2Expected = {0.7f, 0, 0, 0, 0.51f, 0, 0, 0};
    constexpr std::array<float, kBlockSize> kOut3Expected = {0.6f, 0, 0, 0, 0, 0.64f, 0, 0};

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
    std::array<float, kChannelCount> gains = {0.9f, 0.8f, 0.7f, 0.6f};
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

    constexpr std::array<float, kBlockSize> kOut0Expected = {0.810000f,  0.000000f,  -0.171000f, 0.000000f,
                                                             -0.153900f, -0.171000f, -0.138510f, 0.036100f};
    constexpr std::array<float, kBlockSize> kOut1Expected = {0.640000f,  -0.288000f, -0.230400f, -0.184320f,
                                                             -0.435456f, 0.011635f,  0.009308f,  0.007447f};
    constexpr std::array<float, kBlockSize> kOut2Expected = {0.490000f,  0.000000f, 0.000000f,  0.000000f,
                                                             -0.357000f, 0.000000f, -0.357000f, 0.000000f};
    constexpr std::array<float, kBlockSize> kOut3Expected = {0.360000f,  0.000000f,  -0.384000f, 0.000000f,
                                                             -0.230400f, -0.384000f, -0.138240f, 0.409600f};

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
        0.1f,    0.09f,   0.081f,  0.0729f, 0.06561f, 0.059049f, 0.0531441f, 0.04782969f, 0.2000f, 0.1600f, 0.1280f,
        0.1024f, 0.0819f, 0.0655f, 0.0524f, 0.0419f,  0.3000f,   0.2100f,    0.1470f,     0.1029f, 0.0720f, 0.0504f,
        0.0353f, 0.0247f, 0.4000f, 0.2400f, 0.1440f,  0.0864f,   0.0518f,    0.0311f,     0.0187f, 0.0112f};

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
