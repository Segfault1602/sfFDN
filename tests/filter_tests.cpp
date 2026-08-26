#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <limits>

#include <sndfile.h>

#include "sffdn/audio_buffer.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"
#include "rng.h"

namespace
{
constexpr std::array<sfFDN::FilterCoefficients, 11> kTestSOS = {
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

    constexpr uint32_t kBlockSize = 16;
    constexpr uint32_t kSize = kBlockSize * 8;
    std::array<float, kSize> input = {0.f};
    input[0] = 1.f;
    std::array<float, kSize> output{};

    for (auto i = 0u; i < kSize; i += kBlockSize)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, std::span(input).subspan(i, kBlockSize));
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, std::span(output).subspan(i, kBlockSize));

        filter.Process(input_buffer, output_buffer);
    }

    for (auto i = 0u; i < kFirSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(ir[i], 1e-5));
    }

    for (auto i = kFirSize; i < kSize; ++i)
    {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(0.f, 1e-5));
    }

    sfFDN::Fir tick_filter;
    tick_filter.SetCoefficients(ir);
    for (auto i = 0u; i < kFirSize; ++i)
    {
        REQUIRE_THAT(tick_filter.Tick(i == 0 ? 1.f : 0.f), Catch::Matchers::WithinAbs(ir[i], 1e-5));
    }
}

TEST_CASE("SparseFirFilter")
{
    constexpr uint32_t kFirSize = 64;
    std::vector<float> ir(kFirSize, 0.f);
    std::vector<float> sparse_ir;

    sfFDN::SparseFirOptions sparse_fir_config;

    sfFDN::RNG rng;
    for (auto i = 0u; i < kFirSize; i++)
    {
        if (i % 4 == 0)
        {
            auto s = rng();
            ir[i] = s;
            sparse_ir.push_back(s);
            sparse_fir_config.coeffs.push_back({i, s});
        }

    }

    sfFDN::FirOptions fir_config;
    fir_config.coeffs = ir;
    sfFDN::Fir filter(fir_config);

    auto sparse_filter = sfFDN::MakeFirFilter(fir_config, 0.25f);
    // Make sure that the sparse filter is actually created
    REQUIRE(dynamic_cast<sfFDN::SparseFir*>(sparse_filter.get()) != nullptr);

    // sfFDN::SparseFir sparse_filter;
    // sparse_filter.SetCoefficients(sparse_fir_config);

    constexpr uint32_t kSize = 128;
    std::array<float, kSize> input = {0.f};
    input[0] = 1.f;
    std::array<float, kSize> output{};
    std::array<float, kSize> sparse_output{};

    sfFDN::AudioBuffer input_buffer(kSize, 1, input);
    sfFDN::AudioBuffer output_buffer(kSize, 1, output);
    sfFDN::AudioBuffer sparse_output_buffer(kSize, 1, sparse_output);

    filter.Process(input_buffer, output_buffer);
    sparse_filter->Process(input_buffer, sparse_output_buffer);

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

TEST_CASE("SparseFir supports default construction and coefficient updates")
{
    sfFDN::SparseFir filter;

    constexpr uint32_t kBlockSize = 32;
    std::array<float, kBlockSize> input{};
    std::array<float, kBlockSize> empty_output{};
    input[0] = 1.f;
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer empty_output_buffer(empty_output);
    filter.Process(input_buffer, empty_output_buffer);
    REQUIRE(std::ranges::all_of(empty_output, [](float sample) { return sample == 0.f; }));

    sfFDN::SparseFirOptions options;
    options.coeffs = {{0, 0.5f}, {3, -0.25f}, {11, 0.125f}};
    filter.SetCoefficients(options);
    auto clone = filter.Clone();

    std::array<float, kBlockSize> output{};
    std::array<float, kBlockSize> clone_output{};
    sfFDN::AudioBuffer output_buffer(output);
    sfFDN::AudioBuffer clone_output_buffer(clone_output);
    filter.Process(input_buffer, output_buffer);
    clone->Process(input_buffer, clone_output_buffer);

    REQUIRE(output == clone_output);
    REQUIRE_THAT(output[0], Catch::Matchers::WithinAbs(0.5f, 1e-6f));
    REQUIRE_THAT(output[3], Catch::Matchers::WithinAbs(-0.25f, 1e-6f));
    REQUIRE_THAT(output[11], Catch::Matchers::WithinAbs(0.125f, 1e-6f));
}

TEST_CASE("SparseFir continuously consumes streaming blocks")
{
    constexpr uint32_t kFirSize = 64;
    constexpr uint32_t kBlockSize = 128;
    constexpr uint32_t kBlockCount = 40;
    std::vector<float> coefficients(kFirSize, 0.f);
    sfFDN::SparseFirOptions sparse_options;
    for (auto tap = 0u; tap < kFirSize; tap += 4)
    {
        const float coefficient = static_cast<float>(static_cast<int>(tap) - 24) / 64.f;
        coefficients[tap] = coefficient;
        sparse_options.coeffs.emplace_back(tap, coefficient);
    }

    sfFDN::Fir reference({.coeffs = coefficients});
    sfFDN::SparseFir sparse(sparse_options);
    std::array<float, kBlockSize> input{};
    std::array<float, kBlockSize> reference_output{};
    std::array<float, kBlockSize> sparse_output{};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer reference_output_buffer(reference_output);
    sfFDN::AudioBuffer sparse_output_buffer(sparse_output);

    for (auto block = 0u; block < kBlockCount; ++block)
    {
        std::ranges::fill(input, 0.f);
        input[0] = static_cast<float>(block + 1);
        reference.Process(input_buffer, reference_output_buffer);
        sparse.Process(input_buffer, sparse_output_buffer);
        for (auto sample = 0u; sample < kBlockSize; ++sample)
        {
            REQUIRE_THAT(sparse_output[sample], Catch::Matchers::WithinAbs(reference_output[sample], 1e-5f));
        }
    }

    constexpr uint32_t kOversizedBlockSize = 5000;
    sfFDN::Fir oversized_reference({.coeffs = coefficients});
    sfFDN::SparseFir oversized_sparse(sparse_options);
    std::vector<float> oversized_input(kOversizedBlockSize, 0.f);
    std::vector<float> oversized_reference_output(kOversizedBlockSize, 0.f);
    std::vector<float> oversized_sparse_output(kOversizedBlockSize, 0.f);
    for (auto sample = 0u; sample < kOversizedBlockSize; sample += 257)
    {
        oversized_input[sample] = static_cast<float>(sample + 1) / static_cast<float>(kOversizedBlockSize);
    }
    sfFDN::AudioBuffer oversized_input_buffer(oversized_input);
    sfFDN::AudioBuffer oversized_reference_buffer(oversized_reference_output);
    sfFDN::AudioBuffer oversized_sparse_buffer(oversized_sparse_output);
    oversized_reference.Process(oversized_input_buffer, oversized_reference_buffer);

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        oversized_sparse.Process(oversized_input_buffer, oversized_sparse_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);
    for (auto sample = 0u; sample < kOversizedBlockSize; ++sample)
    {
        REQUIRE_THAT(oversized_sparse_output[sample],
                     Catch::Matchers::WithinAbs(oversized_reference_output[sample], 1e-5f));
    }
}

TEST_CASE("SparseFir falls back when tap span and block exceed ring headroom")
{
    constexpr uint32_t kFilterOrder = 4000;
    constexpr uint32_t kBlockSize = 256;
    constexpr uint32_t kBlockCount = 20;
    sfFDN::SparseFirOptions options;
    options.coeffs = {{0, 0.5f}, {127, -0.25f}, {kFilterOrder - 1, 0.125f}};
    sfFDN::SparseFir tick_filter(options);
    sfFDN::SparseFir block_filter(options);

    std::array<float, kBlockSize> input{};
    std::array<float, kBlockSize> expected{};
    std::array<float, kBlockSize> output{};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);

    for (auto block = 0u; block < kBlockCount; ++block)
    {
        for (auto sample = 0u; sample < kBlockSize; ++sample)
        {
            input[sample] = static_cast<float>(block * kBlockSize + sample + 1) / 1000.f;
            expected[sample] = tick_filter.Tick(input[sample]);
        }

        size_t allocations = 0;
        {
            sfFDNTest::ScopedAllocationCounter allocation_counter;
            block_filter.Process(input_buffer, output_buffer);
            allocations = allocation_counter.Count();
        }
        REQUIRE(allocations == 0);
        for (auto sample = 0u; sample < kBlockSize; ++sample)
        {
            REQUIRE_THAT(output[sample], Catch::Matchers::WithinAbs(expected[sample], 1e-5f));
        }
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

TEST_CASE("SchroederAllpassSection preserves parallel mode across clone and move")
{
    const sfFDN::SchroederAllpassSectionOptions options{
        .delays = {5, 7, 11},
        .gains = {0.5f, -0.4f, 0.3f},
        .parallel = true,
    };
    sfFDN::SchroederAllpassSection original(options);
    auto clone = original.Clone();
    sfFDN::SchroederAllpassSection moved(std::move(original));

    constexpr uint32_t kBlockSize = 64;
    std::array<float, kBlockSize> input{};
    input[0] = 1.f;
    std::array<float, kBlockSize> clone_output{};
    std::array<float, kBlockSize> moved_output{};
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer clone_output_buffer(clone_output);
    sfFDN::AudioBuffer moved_output_buffer(moved_output);

    clone->Process(input_buffer, clone_output_buffer);
    moved.Process(input_buffer, moved_output_buffer);
    REQUIRE(clone_output == moved_output);

    sfFDN::SchroederAllpassSection reference(options);
    sfFDN::SchroederAllpassSection in_place(options);
    auto aliased = input;
    std::array<float, kBlockSize> reference_output{};
    sfFDN::AudioBuffer reference_output_buffer(reference_output);
    sfFDN::AudioBuffer aliased_buffer(aliased);
    reference.Process(input_buffer, reference_output_buffer);

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        in_place.Process(aliased_buffer, aliased_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);
    for (auto sample = 0u; sample < kBlockSize; ++sample)
    {
        REQUIRE_THAT(aliased[sample], Catch::Matchers::WithinAbs(reference_output[sample], 1e-5f));
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

    std::array<float, kChannelCount> delays = {2, 3, 4, 5};
    std::array<float, kChannelCount> gains = {-0.9f, -0.8f, -0.7f, -0.6f};

    sfFDN::MultichannelSchroederAllpassSectionOptions options;
    for (auto i = 0u; i < kChannelCount; i++)
    {
        sfFDN::SchroederAllpassSectionOptions section_options;
        section_options.delays = {delays[i]};
        section_options.gains = {gains[i]};
        options.sections.push_back(section_options);
    }

    auto filter = sfFDN::MakeMultichannelSchroederAllpassSection(options);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    filter->Process(input_buffer, output_buffer);

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

    std::array<float, kChannelCount * 2> delays = {2, 5, 4, 1, 4, 6, 2, 5};
    std::array<float, kChannelCount> gains = {0.9f, 0.8f, 0.7f, 0.6f};

    sfFDN::MultichannelSchroederAllpassSectionOptions options;
    for (auto i = 0u; i < kChannelCount; i++)
    {
        sfFDN::SchroederAllpassSectionOptions section_options;
        section_options.delays = {delays[i * 2], delays[i * 2 + 1]};
        section_options.gains = {gains[i], gains[i]};
        options.sections.push_back(section_options);
    }

    auto filter = sfFDN::MakeMultichannelSchroederAllpassSection(options);

    std::vector<float> input(kChannelCount * kBlockSize, 0.f);
    for (uint32_t i = 0; i < kChannelCount; ++i)
    {
        input[i * kBlockSize] = 1.f;
    }

    std::vector<float> output(kChannelCount * kBlockSize, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    filter->Process(input_buffer, output_buffer);

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

    filter.SetCoefficients(kTestSOS);

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
    constexpr uint32_t kChannelCount = 2;
    constexpr uint32_t kStageCount = kTestSOS.size();
    std::vector<sfFDN::FilterCoefficients> coeffs;
    for (auto n = 0; n < kChannelCount; ++n)
    {
        for (auto i = 0u; i < kStageCount; i++)
        {
            coeffs.push_back(kTestSOS[i]);
        }
    }

    sfFDN::IIRFilterBank filter_bank;
    filter_bank.SetFilter(coeffs, kChannelCount);

    constexpr uint32_t kBlockSize = 16;
    std::vector<float> input(kBlockSize * kChannelCount, 0.f);
    std::vector<float> output(kBlockSize * kChannelCount, 0.f);

    sfFDN::AudioBuffer input_buffer(kBlockSize, kChannelCount, input);
    sfFDN::AudioBuffer output_buffer(kBlockSize, kChannelCount, output);

    for (auto i = 0u; i < kChannelCount; ++i)
    {
        input_buffer.GetChannelSpan(i)[0] = 1.f;
    }

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        filter_bank.Process(input_buffer, output_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);

    for (auto i = 0u; i < kBlockSize; ++i)
    {
        for (auto n = 0; n < kChannelCount; ++n)
        {
            REQUIRE_THAT(output_buffer.GetChannelSpan(n)[i],
                         Catch::Matchers::WithinAbs(kTestSOSExpectedOutput[i], 0.0001));
        }
    }

    auto clone = filter_bank.Clone();
    auto clone_input = input;
    std::vector<float> clone_output(output.size(), 0.f);
    sfFDN::AudioBuffer clone_input_buffer(kBlockSize, kChannelCount, clone_input);
    sfFDN::AudioBuffer clone_output_buffer(kBlockSize, kChannelCount, clone_output);
    clone->Process(clone_input_buffer, clone_output_buffer);
    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(output[i], 0.0001));
    }

    filter_bank.Clear();
    auto in_place = input;
    sfFDN::AudioBuffer in_place_buffer(kBlockSize, kChannelCount, in_place);
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        filter_bank.Process(in_place_buffer, in_place_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);

    for (auto i = 0u; i < output.size(); ++i)
    {
        REQUIRE_THAT(in_place[i], Catch::Matchers::WithinAbs(output[i], 0.0001));
    }
}

TEST_CASE("IIRFilterBank preserves channel coefficient layout")
{
    constexpr uint32_t kChannelCount = 2;
    constexpr uint32_t kBlockSize = 4;
    constexpr std::array<sfFDN::FilterCoefficients, 4> kCoefficients = {{{2.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                                                                         {3.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                                                                         {-1.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                                                                         {0.5f, 0.f, 0.f, 1.f, 0.f, 0.f}}};
    std::array<float, kChannelCount * kBlockSize> samples = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};

    sfFDN::IIRFilterBank filter_bank;
    filter_bank.SetFilter(kCoefficients, kChannelCount);

    sfFDN::AudioBuffer buffer(kBlockSize, kChannelCount, samples);
    filter_bank.Process(buffer, buffer);

    constexpr std::array<float, kChannelCount * kBlockSize> kExpected = {6.f,   12.f, 18.f,  24.f,
                                                                         -0.5f, -1.f, -1.5f, -2.f};
    for (auto i = 0u; i < samples.size(); ++i)
    {
        REQUIRE_THAT(samples[i], Catch::Matchers::WithinAbs(kExpected[i], std::numeric_limits<float>::epsilon()));
    }
}

TEST_CASE("IIRFilterBank matches per-channel cascades for every channel count")
{
    // The bank vectorizes across channels in groups of four and splits wide banks into several
    // passes. This walks every channel count across those boundaries, including counts that are
    // not a multiple of the SIMD width, so padding lanes and uneven passes stay covered.
    constexpr uint32_t kBlockSize = 150;
    constexpr uint32_t kStageCount = 5;

    sfFDN::RNG rng;

    for (uint32_t channel_count = 1; channel_count <= 34; ++channel_count)
    {
        std::vector<sfFDN::FilterCoefficients> coeffs;
        coeffs.reserve(channel_count * kStageCount);
        for (uint32_t channel = 0; channel < channel_count; ++channel)
        {
            for (uint32_t stage = 0; stage < kStageCount; ++stage)
            {
                coeffs.push_back({.b0 = 0.6f + (0.2f * rng()),
                                  .b1 = 0.2f * rng(),
                                  .b2 = 0.2f * rng(),
                                  .a0 = 1.f,
                                  .a1 = 0.3f * rng(),
                                  .a2 = 0.2f * rng()});
            }
        }

        std::vector<float> input(channel_count * kBlockSize);
        for (auto& sample : input)
        {
            sample = rng();
        }

        std::vector<float> bank_output(input.size(), 0.f);
        std::vector<float> reference(input.size(), 0.f);

        const sfFDN::AudioBuffer input_buffer(kBlockSize, channel_count, input);
        sfFDN::AudioBuffer bank_buffer(kBlockSize, channel_count, bank_output);
        sfFDN::AudioBuffer reference_buffer(kBlockSize, channel_count, reference);

        sfFDN::IIRFilterBank filter_bank;
        filter_bank.SetFilter(coeffs, channel_count);
        filter_bank.Process(input_buffer, bank_buffer);

        for (uint32_t channel = 0; channel < channel_count; ++channel)
        {
            sfFDN::CascadedBiquads cascade;
            cascade.SetCoefficients(
                std::span<const sfFDN::FilterCoefficients>(coeffs).subspan(channel * kStageCount, kStageCount));

            auto channel_output = reference_buffer.GetChannelBuffer(channel);
            cascade.Process(input_buffer.GetChannelBuffer(channel), channel_output);
        }

        for (size_t i = 0; i < reference.size(); ++i)
        {
            INFO("channel_count = " << channel_count << ", sample " << i);
            REQUIRE_THAT(bank_output[i], Catch::Matchers::WithinAbs(reference[i], 1e-5f));
        }
    }
}
