#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sffdn/sffdn.h>

#include "allocation_counter.h"
#include "filter_design_internal.h"
#include "signal_test_utils.h"

namespace
{
void RequireStableSections(std::span<const sfFDN::FilterCoefficients> sections)
{
    for (const auto& section : sections)
    {
        REQUIRE(std::isfinite(section.b0));
        REQUIRE(std::isfinite(section.b1));
        REQUIRE(std::isfinite(section.b2));
        REQUIRE(std::isfinite(section.a0));
        REQUIRE(std::isfinite(section.a1));
        REQUIRE(std::isfinite(section.a2));
        REQUIRE_THAT(section.a0, Catch::Matchers::WithinAbs(1.f, 1e-6f));

        const std::complex<float> discriminant = std::complex<float>(section.a1 * section.a1 - 4.f * section.a2, 0.f);
        const auto root = std::sqrt(discriminant);
        REQUIRE(std::abs((-section.a1 + root) * 0.5f) < 1.f);
        REQUIRE(std::abs((-section.a1 - root) * 0.5f) < 1.f);
    }
}

float Magnitude(const sfFDN::FilterCoefficients& coefficients, float normalized_frequency)
{
    const std::complex<float> z_inverse =
        std::exp(std::complex<float>(0.F, -2.F * std::numbers::pi_v<float> * normalized_frequency));
    const auto numerator = coefficients.b0 + coefficients.b1 * z_inverse + coefficients.b2 * z_inverse * z_inverse;
    const auto denominator = coefficients.a0 + coefficients.a1 * z_inverse + coefficients.a2 * z_inverse * z_inverse;
    return std::abs(numerator / denominator);
}

void RequireNormalized(const sfFDN::FilterCoefficients& coefficients)
{
    REQUIRE_THAT(coefficients.a0, Catch::Matchers::WithinAbs(1.F, 1e-6F));
    REQUIRE(std::isfinite(coefficients.b0));
    REQUIRE(std::isfinite(coefficients.b1));
    REQUIRE(std::isfinite(coefficients.b2));
    REQUIRE(std::isfinite(coefficients.a1));
    REQUIRE(std::isfinite(coefficients.a2));
}

void RequireSectionsClose(std::span<const sfFDN::FilterCoefficients> actual,
                          std::span<const sfFDN::FilterCoefficients> expected)
{
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const auto normalized = actual[index].Normalize();
        REQUIRE_THAT(normalized.b0, Catch::Matchers::WithinAbs(expected[index].b0, 2e-6F));
        REQUIRE_THAT(normalized.b1, Catch::Matchers::WithinAbs(expected[index].b1, 2e-6F));
        REQUIRE_THAT(normalized.b2, Catch::Matchers::WithinAbs(expected[index].b2, 2e-6F));
        REQUIRE_THAT(normalized.a0, Catch::Matchers::WithinAbs(expected[index].a0, 2e-6F));
        REQUIRE_THAT(normalized.a1, Catch::Matchers::WithinAbs(expected[index].a1, 2e-6F));
        REQUIRE_THAT(normalized.a2, Catch::Matchers::WithinAbs(expected[index].a2, 2e-6F));
    }
}
} // namespace

TEST_CASE("FilterDesigner preserves ten-band reference coefficients", "[filter_design]")
{
    constexpr float kSR = 48000;
    constexpr std::array<double, 10> kT60s = {2.5, 2.7, 2.5, 2.3, 2.3, 2.1, 1.7, 1.6, 1.2, 1.0};
    // constexpr std::array<double, 10> kT60s = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    constexpr float kDelay = 1619;
    constexpr float kShelfCutoff = 8000.0f;

    std::vector<double> coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    // clang-format off
    constexpr std::array<double, 66> kExpectedSOS = {
        0.833753922053558, -0.196112500418774, 0, 1.000000000000000, -0.300074975165566, 0,
        0.999995541761545, -1.995969983773138, 0.995991141545255, 1.000000000000000, -1.995969983773138, 0.995986683306800,
        1.000031455441331, -1.991962851696166, 0.991998061679195, 1.000000000000000, -1.991962851696166, 0.992029517120525,
        1.000000814161373, -1.983794860302083, 0.984059636620207, 1.000000000000000, -1.983794860302083, 0.984060450781580,
        0.999893742170679, -1.967208144896469, 0.968368234838074, 1.000000000000000, -1.967208144896469, 0.968261977008753,
        0.999917777296403, -1.933453811844555, 0.937684589949490, 1.000000000000000, -1.933453811844555, 0.937602367245893,
        0.999700260098036, -1.862502303348346, 0.878873502206346, 1.000000000000000, -1.862502303348346, 0.878573762304382,
        0.996510690312741, -1.706273726254403, 0.769953855737997, 1.000000000000000, -1.706273726254403, 0.766464546050738,
        0.997725098051978, -1.383712253277730, 0.600048185769784, 1.000000000000000, -1.383712253277730, 0.597773283821762,
        0.992215955533026, -0.682207683428299, 0.372199411323571, 1.000000000000000, -0.682207683428299, 0.364415366856597,
        0.995766905324853, 0.598066031393632, 0.200365157462411, 1.000000000000000, 0.598066031393632, 0.196132062787264
    };
    // clang-format on

    for (auto i = 0u; i < coeffs.size(); ++i)
    {
        REQUIRE_THAT(coeffs[i], Catch::Matchers::WithinAbs(kExpectedSOS.at(i), 1e-13));
    }

    std::array<float, 10> t60s_f{};
    for (size_t i = 0; i < kT60s.size(); ++i)
    {
        t60s_f[i] = static_cast<float>(kT60s[i]);
    }

    sfFDN::TenBandFilterOptions config;
    config.t60s = t60s_f;
    config.delay = kDelay;
    config.shelf_cutoff = kShelfCutoff;

    const sfFDN::FilterDesigner designer(kSR);
    const auto float_coeffs = designer.DesignFilter(config);
    for (auto i = 0u; i < float_coeffs.size(); ++i)
    {
        REQUIRE_THAT(float_coeffs[i].b0, Catch::Matchers::WithinAbs(kExpectedSOS.at(i * 6), 1e-7));
    }
}

TEST_CASE("Polyval matches a complex polynomial reference", "[filter_design]")
{
    constexpr size_t kN = 10;
    std::array<double, kN> freqs = {31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
    std::array<std::complex<double>, kN> dig_w;

    for (auto [w, f] : std::views::zip(dig_w, freqs))
    {
        w = std::exp(std::complex<double>(0.0, 1.0) * f);
    }

    std::array p = {0.5, -0.8, 0.2};

    std::array<std::complex<double>, kN> result;
    sfFDN::Polyval<double>(p, dig_w, result);

    std::array<std::complex<double>, kN> expected = {std::complex<double>(-0.116292474735830, -0.030764807808418),
                                                     std::complex<double>(-0.162494939592148, -0.047383785262679),
                                                     std::complex<double>(-0.309677457072758, +0.007568357580022),
                                                     std::complex<double>(-0.434715280943946, +0.542536512972206),
                                                     std::complex<double>(1.188268956890534, +0.787657214523982),
                                                     std::complex<double>(-0.433633035582978, -0.196483880217534),
                                                     std::complex<double>(0.128994159506051, -1.085783500471624),
                                                     std::complex<double>(0.816780131394544, +1.045724551283134),
                                                     std::complex<double>(-0.348206819242412, -0.732770892795190),
                                                     std::complex<double>(1.475942296183083, -0.234683626155909)};

    for (const auto [res, exp] : std::views::zip(result, expected))
    {
        REQUIRE_THAT(res.imag(), Catch::Matchers::WithinAbs(exp.imag(), 1e-14));
        REQUIRE_THAT(res.real(), Catch::Matchers::WithinAbs(exp.real(), 1e-14));
    }
}

TEST_CASE("FilterDesigner produces stable graphic EQ sections", "[filter_design]")
{
    constexpr std::array<float, 10> kFreq = {31.25f, 62.5f,  125.f,  250.f,  500.f,
                                             1000.f, 2000.f, 4000.f, 8000.f, 16000.f};
    constexpr std::array<float, 10> kMag = {-3.f, -2.f, -1.f, 0.5f, 1.f, 0.75f, -0.5f, 1.f, -1.f, -3.f};

    const sfFDN::FilterDesigner designer(48000.F);
    const auto sections = designer.DesignFilter(sfFDN::GraphicEQOptions{.gains_db = kMag, .freqs = kFreq});
    RequireStableSections(sections);
}

TEST_CASE("FilterDesigner produces stable three-band sections", "[filter_design]")
{
    constexpr float kDelay = 1000.f;
    constexpr float sr = 48000.f;
    sfFDN::ThreeBandFilterOptions config{{2.f, 1.f, 0.5f}, kDelay, {300.f, 8000.f}, 1.f / std::sqrt(2.f)};

    const sfFDN::FilterDesigner designer(sr);
    const auto sos = designer.DesignFilter(config);
    REQUIRE(sos.size() == 2);
    RequireStableSections(sos);
}

TEST_CASE("CreateAttenuationFilterBank selects multichannel cascades and falls back for heterogeneous filters",
          "[filter_design]")
{
    sfFDN::AttenuationFilterBankOptions ten_band_options;
    ten_band_options.filter_configs.emplace_back(sfFDN::TenBandFilterOptions{
        .t60s = {2.f, 2.f, 1.8f, 1.6f, 1.4f, 1.2f, 1.f, 0.8f, 0.6f, 0.5f},
        .delay = 1000.f,
        .shelf_cutoff = 8000.f,
    });
    ten_band_options.filter_configs.emplace_back(sfFDN::TenBandFilterOptions{
        .t60s = {1.8f, 1.7f, 1.6f, 1.5f, 1.4f, 1.3f, 1.2f, 1.1f, 1.f, 0.9f},
        .delay = 1200.f,
        .shelf_cutoff = 8000.f,
    });

    const sfFDN::FilterDesigner designer(48000.F);
    auto optimized = sfFDN::CreateAttenuationFilterBank(ten_band_options, designer);
    REQUIRE(dynamic_cast<sfFDN::IIRFilterBank*>(optimized.get()) != nullptr);

    auto reference = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : ten_band_options.filter_configs)
    {
        reference->AddFilter(sfFDN::CreateAttenuationFilter(config, designer));
    }

    constexpr uint32_t kSampleCount = 64;
    std::array<float, 2 * kSampleCount> optimized_output{};
    for (auto i = 0u; i < optimized_output.size(); ++i)
    {
        optimized_output[i] = static_cast<float>(static_cast<int>((i * 29u) % 79u) - 39) / 39.f;
    }
    auto reference_output = optimized_output;
    sfFDN::AudioBuffer optimized_buffer(kSampleCount, 2, optimized_output);
    sfFDN::AudioBuffer reference_buffer(kSampleCount, 2, reference_output);
    optimized->Process(optimized_buffer, optimized_buffer);
    reference->Process(reference_buffer, reference_buffer);
    for (auto i = 0u; i < optimized_output.size(); ++i)
    {
        const float tolerance = 1e-6f + std::abs(reference_output[i]) * 5e-3f;
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], tolerance));
    }

    auto clone = optimized->Clone();
    optimized->Clear();
    optimized_output = reference_output;
    auto clone_output = reference_output;
    sfFDN::AudioBuffer clone_buffer(kSampleCount, 2, clone_output);
    optimized->Process(optimized_buffer, optimized_buffer);
    clone->Process(clone_buffer, clone_buffer);
    REQUIRE(optimized_output == clone_output);

    auto heterogeneous_options = ten_band_options;
    heterogeneous_options.filter_configs[1] = sfFDN::TwoBandFilterOptions{.t60s = {1.5f, 0.7f}, .delay = 1200.f};
    auto fallback = sfFDN::CreateAttenuationFilterBank(heterogeneous_options, designer);
    REQUIRE(dynamic_cast<sfFDN::FilterBank*>(fallback.get()) != nullptr);
}

TEST_CASE("CreateAttenuationFilterBank matches three-band channel filters without allocations", "[filter_design]")
{
    sfFDN::AttenuationFilterBankOptions options;
    options.filter_configs.emplace_back(sfFDN::ThreeBandFilterOptions{
        .t60s = {1.5f, 1.f, 0.5f},
        .delay = 1000.f,
        .freqs = {800.f, 8000.f},
    });
    options.filter_configs.emplace_back(sfFDN::ThreeBandFilterOptions{
        .t60s = {1.8f, 1.2f, 0.7f},
        .delay = 1200.f,
        .freqs = {600.f, 6000.f},
    });

    const sfFDN::FilterDesigner designer(48000.F);
    auto optimized = sfFDN::CreateAttenuationFilterBank(options, designer);
    REQUIRE(dynamic_cast<sfFDN::IIRFilterBank*>(optimized.get()) != nullptr);
    auto clone = optimized->Clone();

    auto reference = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : options.filter_configs)
    {
        reference->AddFilter(sfFDN::CreateAttenuationFilter(config, designer));
    }

    constexpr uint32_t kSampleCount = 128;
    std::array<float, 2 * kSampleCount> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(static_cast<int>((i * 31u) % 83u) - 41) / 41.f;
    }

    auto optimized_output = input;
    auto clone_output = input;
    auto reference_output = input;
    sfFDN::AudioBuffer optimized_buffer(kSampleCount, 2, optimized_output);
    sfFDN::AudioBuffer clone_buffer(kSampleCount, 2, clone_output);
    sfFDN::AudioBuffer reference_buffer(kSampleCount, 2, reference_output);

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        optimized->Process(optimized_buffer, optimized_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);
    clone->Process(clone_buffer, clone_buffer);
    reference->Process(reference_buffer, reference_buffer);

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
        REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
    }

    optimized->Clear();
    optimized_output = input;
    optimized->Process(optimized_buffer, optimized_buffer);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
    }

    optimized->Clear();
    reference->Clear();
    std::vector<float> sustained_optimized;
    std::vector<float> sustained_reference;
    sustained_optimized.reserve(375 * input.size());
    sustained_reference.reserve(375 * input.size());
    for (auto block = 0u; block < 375; ++block)
    {
        for (auto i = 0u; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(static_cast<int>(((block * input.size() + i) * 31u) % 83u) - 41) / 41.f;
        }
        optimized_output = input;
        reference_output = input;
        optimized->Process(optimized_buffer, optimized_buffer);
        reference->Process(reference_buffer, reference_buffer);
        sustained_optimized.insert(sustained_optimized.end(), optimized_output.begin(), optimized_output.end());
        sustained_reference.insert(sustained_reference.end(), reference_output.begin(), reference_output.end());
    }

    sfFDNTest::RequireSignalsClose(sustained_reference, sustained_optimized, 3e-5f, 90.0);
}

TEST_CASE("CreateAttenuationFilterBank matches two-band channel filters across platform implementations",
          "[filter_design]")
{
    sfFDN::AttenuationFilterBankOptions options;
    options.filter_configs.emplace_back(sfFDN::TwoBandFilterOptions{.t60s = {1.5f, 0.5f}, .delay = 1000.f});
    options.filter_configs.emplace_back(sfFDN::TwoBandFilterOptions{.t60s = {1.8f, 0.7f}, .delay = 1200.f});

    const sfFDN::FilterDesigner designer(48000.F);
    auto optimized = sfFDN::CreateAttenuationFilterBank(options, designer);
#if defined(__APPLE__) && defined(__aarch64__) && defined(SFFDN_USE_VDSP)
    REQUIRE(dynamic_cast<sfFDN::IIRFilterBank*>(optimized.get()) != nullptr);
#else
    REQUIRE(dynamic_cast<sfFDN::FilterBank*>(optimized.get()) != nullptr);
#endif
    auto clone = optimized->Clone();

    auto reference = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : options.filter_configs)
    {
        reference->AddFilter(sfFDN::CreateAttenuationFilter(config, designer));
    }

    constexpr uint32_t kSampleCount = 128;
    std::array<float, 2 * kSampleCount> input{};
    for (auto i = 0u; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(static_cast<int>((i * 43u) % 97u) - 48) / 48.f;
    }
    auto optimized_output = input;
    auto clone_output = input;
    auto reference_output = input;
    sfFDN::AudioBuffer optimized_buffer(kSampleCount, 2, optimized_output);
    sfFDN::AudioBuffer clone_buffer(kSampleCount, 2, clone_output);
    sfFDN::AudioBuffer reference_buffer(kSampleCount, 2, reference_output);

    size_t allocations = 0;
    {
        sfFDNTest::ScopedAllocationCounter allocation_counter;
        optimized->Process(optimized_buffer, optimized_buffer);
        allocations = allocation_counter.Count();
    }
    REQUIRE(allocations == 0);
    clone->Process(clone_buffer, clone_buffer);
    reference->Process(reference_buffer, reference_buffer);

    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
        REQUIRE_THAT(clone_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
    }

    optimized->Clear();
    optimized_output = input;
    optimized->Process(optimized_buffer, optimized_buffer);
    for (auto i = 0u; i < input.size(); ++i)
    {
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], 2e-5f));
    }

    optimized->Clear();
    reference->Clear();
    std::vector<float> sustained_optimized;
    std::vector<float> sustained_reference;
    sustained_optimized.reserve(375 * input.size());
    sustained_reference.reserve(375 * input.size());
    for (auto block = 0u; block < 375; ++block)
    {
        for (auto i = 0u; i < input.size(); ++i)
        {
            input[i] = static_cast<float>(static_cast<int>(((block * input.size() + i) * 43u) % 97u) - 48) / 48.f;
        }
        optimized_output = input;
        reference_output = input;
        optimized->Process(optimized_buffer, optimized_buffer);
        reference->Process(reference_buffer, reference_buffer);
        sustained_optimized.insert(sustained_optimized.end(), optimized_output.begin(), optimized_output.end());
        sustained_reference.insert(sustained_reference.end(), reference_output.begin(), reference_output.end());
    }

    sfFDNTest::RequireSignalsClose(sustained_reference, sustained_optimized, 3e-5f, 90.0);
}

TEST_CASE("FilterDesigner matches homogeneous and two-band reference values", "[filter_design]")
{
    struct Reference
    {
        float sample_rate;
        float homogeneous_gain;
        std::array<float, 2> two_band;
    };
    constexpr std::array references = {
        Reference{44100.F, 0.892469525F, {0.835457802F, -0.0638956055F}},
        Reference{48000.F, 0.900783122F, {0.847880006F, -0.0587165467F}},
        Reference{96000.F, 0.949096978F, {0.921201468F, -0.0293836035F}},
    };

    const sfFDN::HomogenousFilterOptions homogeneous{.t60 = 1.7F, .delay = 1234.5F};
    const sfFDN::TwoBandFilterOptions two_band{.t60s = {1.7F, 0.8F}, .delay = 1234.5F};

    for (const auto& reference : references)
    {
        const sfFDN::FilterDesigner designer(reference.sample_rate);
        REQUIRE_THAT(designer.DesignFilter(homogeneous), Catch::Matchers::WithinAbs(reference.homogeneous_gain, 1e-6F));
        const auto actual_two_band = designer.DesignFilter(two_band);
        REQUIRE_THAT(actual_two_band.first, Catch::Matchers::WithinAbs(reference.two_band[0], 1e-6F));
        REQUIRE_THAT(actual_two_band.second, Catch::Matchers::WithinAbs(reference.two_band[1], 1e-6F));
    }
}

TEST_CASE("FilterDesigner matches complete SOS reference coefficients", "[filter_design]")
{
    using Coefficients = sfFDN::FilterCoefficients;
    struct Reference
    {
        float sample_rate;
        std::array<Coefficients, 2> three_band;
        std::array<Coefficients, 11> ten_band;
        std::array<Coefficients, 11> graphic_eq;
    };
    // Normalized reference coefficient order: {b0, b1, b2, a0, a1, a2}.
    constexpr std::array references = {
        Reference{44100.F,
                  {{{0.840366006F, -1.57766831F, 0.743267059F, 1.F, -1.88109243F, 0.887775719F},
                    {0.900692821F, -0.739744306F, 0.264301032F, 1.F, -0.889995933F, 0.31524542F}}},
                  {{{0.821527481F, -0.218016908F, 0.F, 1.F, -0.323785216F, 0.F},
                    {1.00000429F, -1.9956218F, 0.995637357F, 1.F, -1.9956218F, 0.995641649F},
                    {0.999982059F, -1.99119687F, 0.991293728F, 1.F, -1.99119687F, 0.991275787F},
                    {0.999926746F, -1.98227489F, 0.982662559F, 1.F, -1.98227489F, 0.982589245F},
                    {0.999762356F, -1.96413529F, 0.965619504F, 1.F, -1.96413529F, 0.965381861F},
                    {0.999329388F, -1.92679787F, 0.93236798F, 1.F, -1.92679787F, 0.931697369F},
                    {0.998265326F, -1.84822404F, 0.86887759F, 1.F, -1.84822404F, 0.867142916F},
                    {0.995826602F, -1.67704856F, 0.751690209F, 1.F, -1.67704856F, 0.747516811F},
                    {0.992562711F, -1.31591082F, 0.570363879F, 1.F, -1.31591082F, 0.56292659F},
                    {0.994843662F, -0.549999595F, 0.321669251F, 1.F, -0.549999595F, 0.316512883F},
                    {0.997585416F, 0.734589279F, 0.130926073F, 1.F, 0.734589279F, 0.128511474F}}},
                  {{{1.05147207F, -0.342100263F, 0.F, 1.F, -0.10695383F, 0.F},
                    {0.99971664F, -1.99532819F, 0.995631278F, 1.F, -1.99532819F, 0.995347917F},
                    {1.00205243F, -1.99308252F, 0.991109073F, 1.F, -1.99308252F, 0.9931615F},
                    {0.99646467F, -1.97858191F, 0.982431054F, 1.F, -1.97858191F, 0.978895724F},
                    {1.0101558F, -1.97327638F, 0.964372993F, 1.F, -1.97327638F, 0.97452873F},
                    {0.992406845F, -1.91982687F, 0.9323017F, 1.F, -1.91982687F, 0.924708545F},
                    {1.04591203F, -1.88701236F, 0.860416174F, 1.F, -1.88701236F, 0.906328261F},
                    {0.96224606F, -1.64705539F, 0.754017293F, 1.F, -1.64705539F, 0.716263354F},
                    {1.08484268F, -1.37300873F, 0.545899987F, 1.F, -1.37300873F, 0.63074261F},
                    {0.927257955F, -0.530552626F, 0.342705518F, 1.F, -0.530552626F, 0.269963503F},
                    {1.02853763F, 0.745848119F, 0.117270157F, 1.F, 0.745848119F, 0.145807818F}}}},
        Reference{48000.F,
                  {{{0.85220629F, -1.60849249F, 0.761403978F, 1.F, -1.89059865F, 0.896280408F},
                    {0.905729234F, -0.826572657F, 0.292411208F, 1.F, -0.972420514F, 0.34398827F}}},
                  {{{0.832569301F, -0.260756165F, 0.F, 1.F, -0.365195274F, 0.F},
                    {1.00000358F, -1.995978F, 0.995991111F, 1.F, -1.995978F, 0.995994747F},
                    {0.99998486F, -1.99191654F, 0.991998255F, 1.F, -1.99191654F, 0.991983175F},
                    {0.999938071F, -1.98373258F, 0.984060049F, 1.F, -1.98373258F, 0.98399812F},
                    {0.999799132F, -1.96711469F, 0.968369424F, 1.F, -1.96711469F, 0.968168497F},
                    {0.999432087F, -1.93298244F, 0.937697947F, 1.F, -1.93298244F, 0.937129974F},
                    {0.99852699F, -1.8614037F, 0.878938675F, 1.F, -1.8614037F, 0.877465606F},
                    {0.996437132F, -1.70621014F, 0.769961596F, 1.F, -1.70621014F, 0.766398728F},
                    {0.993693233F, -1.38089752F, 0.600829959F, 1.F, -1.38089752F, 0.594523191F},
                    {0.995527029F, -0.683349669F, 0.371172279F, 1.F, -0.683349669F, 0.366699338F},
                    {0.997387588F, 0.59855324F, 0.199718848F, 1.F, 0.59855324F, 0.197106466F}}},
                  {{{1.06337285F, -0.394694805F, 0.F, 1.F, -0.158184275F, 0.F},
                    {0.999739647F, -1.99570847F, 0.995985448F, 1.F, -1.99570847F, 0.995725095F},
                    {1.00188613F, -1.99364865F, 0.991829216F, 1.F, -1.99364865F, 0.993715346F},
                    {0.99674952F, -1.98032928F, 0.983844876F, 1.F, -1.98032928F, 0.980594397F},
                    {1.00933838F, -1.97551548F, 0.967235327F, 1.F, -1.97551548F, 0.976573706F},
                    {0.993016124F, -1.92650342F, 0.937620938F, 1.F, -1.92650342F, 0.930637002F},
                    {1.04226506F, -1.89725018F, 0.871356487F, 1.F, -1.89725018F, 0.913621545F},
                    {0.965218008F, -1.67787373F, 0.771844685F, 1.F, -1.67787373F, 0.737062693F},
                    {1.0772109F, -1.43513143F, 0.579936147F, 1.F, -1.43513143F, 0.65714705F},
                    {0.932650805F, -0.660886586F, 0.389122367F, 1.F, -0.660886586F, 0.321773142F},
                    {1.03087711F, 0.608461201F, 0.186045185F, 1.F, 0.608461201F, 0.216922358F}}}},
        Reference{96000.F,
                  {{{0.92278564F, -1.79396987F, 0.872588336F, 1.F, -1.94487178F, 0.946350932F},
                    {0.943543315F, -1.36388135F, 0.536720693F, 1.F, -1.46287322F, 0.579255879F}}},
                  {{{0.904908836F, -0.554477632F, 0.F, 1.F, -0.630770981F, 0.F},
                    {1.00000095F, -1.99799025F, 0.997993588F, 1.F, -1.99799025F, 0.997994483F},
                    {0.999996185F, -1.99597061F, 0.99599117F, 1.F, -1.99597061F, 0.995987356F},
                    {0.999984443F, -1.99191606F, 0.991998255F, 1.F, -1.99191606F, 0.991982758F},
                    {0.999949396F, -1.98374379F, 0.984059989F, 1.F, -1.98374379F, 0.984009385F},
                    {0.999855399F, -1.96717036F, 0.968368709F, 1.F, -1.96717036F, 0.968224168F},
                    {0.999619663F, -1.93316483F, 0.937693119F, 1.F, -1.93316483F, 0.937312722F},
                    {0.999052405F, -1.86189663F, 0.878910422F, 1.F, -1.86189663F, 0.877962828F},
                    {0.998350263F, -1.72207272F, 0.784470618F, 1.F, -1.72207272F, 0.78282088F},
                    {0.998624802F, -1.43396938F, 0.657180429F, 1.F, -1.43396938F, 0.65580523F},
                    {0.99865979F, -0.783670127F, 0.568680465F, 1.F, -0.783670127F, 0.567340255F}}},
                  {{{1.14173532F, -0.741012037F, 0.F, 1.F, -0.49551937F, 0.F},
                    {0.999869704F, -1.99785614F, 0.997990608F, 1.F, -1.99785614F, 0.997860312F},
                    {1.0009445F, -1.99683595F, 0.995908201F, 1.F, -1.99683595F, 0.996852696F},
                    {0.998367667F, -1.99018455F, 0.991883457F, 1.F, -1.99018455F, 0.990251124F},
                    {1.00469267F, -1.98795092F, 0.983524323F, 1.F, -1.98795092F, 0.988216996F},
                    {0.996476114F, -1.96370482F, 0.968280673F, 1.F, -1.96370482F, 0.964756787F},
                    {1.02142894F, -1.95168698F, 0.934445679F, 1.F, -1.95168698F, 0.955874622F},
                    {0.982077003F, -1.84517586F, 0.87902081F, 1.F, -1.84517586F, 0.861097813F},
                    {1.03809845F, -1.75424588F, 0.778030515F, 1.F, -1.75424588F, 0.816128969F},
                    {0.964347064F, -1.40838504F, 0.661915898F, 1.F, -1.40838504F, 0.626262963F},
                    {1.02524114F, -0.793883622F, 0.562526107F, 1.F, -0.793883622F, 0.587767303F}}}},
    };

    const sfFDN::ThreeBandFilterOptions three_band{
        .t60s = {1.7F, 1.1F, 0.6F}, .delay = 1234.5F, .freqs = {600.F, 6000.F}, .q = 0.70710677F};
    const sfFDN::TenBandFilterOptions ten_band{
        .t60s = {1.7F, 1.6F, 1.5F, 1.4F, 1.3F, 1.2F, 1.1F, 1.F, 0.9F, 0.8F}, .delay = 1234.5F, .shelf_cutoff = 7000.F};
    const sfFDN::GraphicEQOptions graphic_eq{
        .gains_db = {-2.F, 1.F, -3.F, 2.F, -1.F, 3.F, -2.F, 1.F, -1.F, 2.F},
        .freqs = {31.25F, 62.5F, 125.F, 250.F, 500.F, 1000.F, 2000.F, 4000.F, 8000.F, 16000.F},
    };
    for (const auto& reference : references)
    {
        const sfFDN::FilterDesigner designer(reference.sample_rate);
        RequireSectionsClose(designer.DesignFilter(three_band), reference.three_band);
        RequireSectionsClose(designer.DesignFilter(ten_band), reference.ten_band);
        RequireSectionsClose(designer.DesignFilter(graphic_eq), reference.graphic_eq);
    }
}

TEST_CASE("FilterDesigner converts T60 to analytic gain", "[filter_design]")
{
    const sfFDN::FilterDesigner designer(48000.F);
    REQUIRE(designer.T60ToGain(1.7F, 0.F) == 1.F);
    REQUIRE_THAT(designer.T60ToGain(1.7F, 1.7F * designer.GetSampleRate()), Catch::Matchers::WithinAbs(0.001F, 1e-7F));
    REQUIRE_THAT(designer.T60ToGain(2.5F, 1234.5F),
                 Catch::Matchers::WithinAbs(std::pow(10.F, -3.F * 1234.5F / (2.5F * 48000.F)), 1e-7F));
    REQUIRE_THROWS_AS(designer.T60ToGain(0.F, 1.F), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.T60ToGain(-1.F, 1.F), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.T60ToGain(1.F, -1.F), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::FilterDesigner(0.F), std::invalid_argument);
}

TEST_CASE("FilterDesigner designs normalized RBJ filters", "[filter_design]")
{
    const sfFDN::FilterDesigner designer(48000.F);
    const auto low_boost = designer.DesignFilter(sfFDN::LowShelfOptions{.frequency = 1000.F, .gain_db = 6.F});
    const auto low_cut = designer.DesignFilter(sfFDN::LowShelfOptions{.frequency = 1000.F, .gain_db = -6.F});
    const auto high_boost = designer.DesignFilter(sfFDN::HighShelfOptions{.frequency = 1000.F, .gain_db = 6.F});
    const auto high_cut = designer.DesignFilter(sfFDN::HighShelfOptions{.frequency = 1000.F, .gain_db = -6.F});
    const auto peak_boost = designer.DesignFilter(sfFDN::PeakingOptions{.frequency = 1000.F, .gain_db = 6.F});
    const auto peak_cut = designer.DesignFilter(sfFDN::PeakingOptions{.frequency = 1000.F, .gain_db = -6.F});
    const auto low_unity = designer.DesignFilter(sfFDN::LowShelfOptions{.frequency = 1000.F, .gain_db = 0.F});
    const auto high_unity = designer.DesignFilter(sfFDN::HighShelfOptions{.frequency = 1000.F, .gain_db = 0.F});

    for (const auto coefficients : {low_boost, low_cut, high_boost, high_cut, peak_boost, peak_cut})
    {
        RequireNormalized(coefficients);
        RequireStableSections(std::span(&coefficients, 1U));
    }
    REQUIRE_THAT(Magnitude(low_boost, 0.F), Catch::Matchers::WithinRel(std::pow(10.F, 6.F / 20.F), 1e-4F));
    REQUIRE_THAT(Magnitude(low_boost, 0.5F), Catch::Matchers::WithinAbs(1.F, 1e-5F));
    REQUIRE_THAT(Magnitude(high_boost, 0.F), Catch::Matchers::WithinAbs(1.F, 1e-4F));
    REQUIRE_THAT(Magnitude(high_boost, 0.5F), Catch::Matchers::WithinRel(std::pow(10.F, 6.F / 20.F), 1e-4F));
    REQUIRE_THAT(Magnitude(peak_boost, 1000.F / 48000.F),
                 Catch::Matchers::WithinRel(std::pow(10.F, 6.F / 20.F), 1e-4F));
    REQUIRE_THAT(Magnitude(low_unity, 0.F), Catch::Matchers::WithinAbs(1.F, 1e-5F));
    REQUIRE_THAT(Magnitude(high_unity, 0.5F), Catch::Matchers::WithinAbs(1.F, 1e-5F));
    REQUIRE_THAT(Magnitude(designer.DesignFilter(sfFDN::PeakingOptions{}), 0.F),
                 Catch::Matchers::WithinAbs(1.F, 1e-5F));
    REQUIRE_THAT(Magnitude(designer.DesignFilter(sfFDN::PeakingOptions{}), 0.5F),
                 Catch::Matchers::WithinAbs(1.F, 1e-5F));
    REQUIRE_THAT(Magnitude(low_boost, 0.F) * Magnitude(low_cut, 0.F), Catch::Matchers::WithinRel(1.F, 1e-4F));
    REQUIRE_THAT(Magnitude(high_boost, 0.5F) * Magnitude(high_cut, 0.5F), Catch::Matchers::WithinRel(1.F, 1e-4F));
    REQUIRE_THAT(Magnitude(peak_boost, 1000.F / 48000.F) * Magnitude(peak_cut, 1000.F / 48000.F),
                 Catch::Matchers::WithinRel(1.F, 1e-4F));
    for (const auto coefficients : {peak_boost, peak_cut})
    {
        REQUIRE_THAT(Magnitude(coefficients, 0.F), Catch::Matchers::WithinAbs(1.F, 1e-4F));
        REQUIRE_THAT(Magnitude(coefficients, 0.5F), Catch::Matchers::WithinAbs(1.F, 1e-5F));
    }
    for (const float frequency : {100.F, 500.F, 2000.F, 10000.F})
    {
        const float normalized = frequency / designer.GetSampleRate();
        REQUIRE_THAT(Magnitude(peak_boost, normalized) * Magnitude(peak_cut, normalized),
                     Catch::Matchers::WithinRel(1.F, 1e-4F));
    }
    const auto narrow_peak =
        designer.DesignFilter(sfFDN::PeakingOptions{.frequency = 1000.F, .gain_db = 6.F, .q = 2.F});
    REQUIRE(Magnitude(narrow_peak, 500.F / designer.GetSampleRate()) <
            Magnitude(peak_boost, 500.F / designer.GetSampleRate()));

    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::PeakingOptions{.frequency = 0.F}), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::PeakingOptions{.frequency = 24000.F}), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::PeakingOptions{.q = 0.F}), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::LowShelfOptions{.frequency = 0.F}), std::invalid_argument);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::HighShelfOptions{.frequency = 24000.F}), std::invalid_argument);
}

TEST_CASE("FilterDesigner rejects non-finite RBJ coefficients", "[filter_design]")
{
    const sfFDN::FilterDesigner designer(48000.F);
    constexpr float kGain = std::numeric_limits<float>::max();
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::LowShelfOptions{.gain_db = kGain}), std::runtime_error);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::HighShelfOptions{.gain_db = kGain}), std::runtime_error);
    REQUIRE_THROWS_AS(designer.DesignFilter(sfFDN::PeakingOptions{.gain_db = kGain}), std::runtime_error);
}

TEST_CASE("CreateAttenuationFilter retains designed coefficients after designer destruction", "[filter_design]")
{
    std::unique_ptr<sfFDN::AudioProcessor> filter;
    {
        const sfFDN::FilterDesigner designer(48000.F);
        filter = sfFDN::CreateAttenuationFilter(sfFDN::HomogenousFilterOptions{.t60 = 1.F, .delay = 48000.F}, designer);
    }

    std::array<float, 1> input{1.F};
    std::array<float, 1> output{};
    const sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    filter->Process(input_buffer, output_buffer);
    REQUIRE_THAT(output[0], Catch::Matchers::WithinAbs(0.001F, 1e-6F));
    auto clone = filter->Clone();
    clone->Clear();
    output[0] = 0.F;
    clone->Process(input_buffer, output_buffer);
    REQUIRE_THAT(output[0], Catch::Matchers::WithinAbs(0.001F, 1e-6F));
}
