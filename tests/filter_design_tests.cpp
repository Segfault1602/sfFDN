#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <print>
#include <ranges>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sffdn/sffdn.h>

#include "allocation_counter.h"
#include "filter_design_internal.h"

TEST_CASE("TwoFilter")
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
    config.sample_rate = kSR;
    config.shelf_cutoff = kShelfCutoff;

    auto float_coeffs = sfFDN::DesignTenBandAbsorption(config);
    for (auto i = 0u; i < float_coeffs.size(); ++i)
    {
        REQUIRE_THAT(float_coeffs[i].b0, Catch::Matchers::WithinAbs(kExpectedSOS.at(i * 6), 1e-7));
    }
}

TEST_CASE("TwoFilter2")
{
    constexpr float kSR = 48000;
    constexpr std::array<double, 10> kT60s = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0};
    // constexpr std::array<double, 10> kT60s = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    constexpr float kDelay = 1619;
    constexpr float kShelfCutoff = 8000.0f;

    std::vector<double> coeffs = sfFDN::GetTwoFilter_d(kT60s, kDelay, kSR, kShelfCutoff);

    for (auto i = 0u; i < coeffs.size(); ++i)
    {
        std::cout << std::setprecision(4) << coeffs[i] << ", ";
        if ((i + 1) % 6 == 0)
        {
            std::cout << "\n";
        }
    }
}

TEST_CASE("Polyval")
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

TEST_CASE("GraphicEQ")
{
    SKIP();
    constexpr double kSR = 48000;
    constexpr double kF0 = 1000.0;
    constexpr double kQ = 0.707;
    constexpr double kDbGain = -6.0;
    constexpr double kWc = kF0 / kSR;

    auto coeffs = sfFDN::LowShelfRBJ(kWc, kDbGain, kQ);

    constexpr std::array<double, 6> kLowShelfExpected = {0.968460511117436f, -1.786264176544932f, 0.828701928321781f,
                                                         1.000000000000000f, -1.780840861366279f, 0.802585754617870f};

    for (auto i = 0u; i < coeffs.size(); ++i)
    {
        REQUIRE_THAT(coeffs[i], Catch::Matchers::WithinAbs(kLowShelfExpected.at(i), 1e-6));
    }

    coeffs = sfFDN::HighShelfRBJ(kWc, kDbGain, kQ);

    constexpr std::array<double, 6> kHighShelfExpected = {0.517509209589753f, -0.921601546570798f, 0.415345519500289f,
                                                          1.000000000000000f, -1.844436769532185f, 0.855689952051429f};

    for (auto i = 0u; i < coeffs.size(); ++i)
    {
        REQUIRE_THAT(coeffs[i], Catch::Matchers::WithinAbs(kHighShelfExpected.at(i), 1e-6));
    }

    constexpr std::array<float, 10> kFreq = {62.5, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 8000};
    constexpr std::array<float, 10> kMag = {1.0, 1.5, 2.0, 0.5, 1.0, 0.9, -0.5, 1.0, -1.0, -6.0};

    auto graphic_eq_coeffs = sfFDN::DesignGraphicEQ({kMag, kFreq, kSR});

    for (auto i = 0u; i < graphic_eq_coeffs.size(); ++i)
    {
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].b0 << ", ";
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].b1 << ", ";
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].b2 << ", ";
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].a0 << ", ";
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].a1 << ", ";
        std::cout << std::setprecision(15) << graphic_eq_coeffs[i].a2 << "\n";
    }
}

TEST_CASE("ThreeBandFilter")
{
    constexpr float kDelay = 1000.f;
    constexpr float sr = 48000.f;
    sfFDN::ThreeBandFilterOptions config{{2.f, 1.f, 0.5f}, kDelay, {300.f, 8000.f}, 1.f / std::sqrt(2.f), sr};

    auto sos = sfFDN::DesignThreeBandAbsorption(config);

    for (auto i = 0u; i < sos.size(); ++i)
    {
        std::cout << std::setprecision(3) << sos[i].b0 << ", " << std::setprecision(3) << sos[i].b1 << ", "
                  << std::setprecision(3) << sos[i].b2 << ", " << std::setprecision(3) << sos[i].a0 << ", "
                  << std::setprecision(3) << sos[i].a1 << ", " << std::setprecision(3) << sos[i].a2 << "\n";
    }
}

TEST_CASE("Attenuation filter bank selects multichannel cascades only when supported")
{
    sfFDN::AttenuationFilterBankOptions ten_band_options;
    ten_band_options.filter_configs.emplace_back(sfFDN::TenBandFilterOptions{
        .t60s = {2.f, 2.f, 1.8f, 1.6f, 1.4f, 1.2f, 1.f, 0.8f, 0.6f, 0.5f},
        .delay = 1000.f,
        .sample_rate = 48000.f,
        .shelf_cutoff = 8000.f,
    });
    ten_band_options.filter_configs.emplace_back(sfFDN::TenBandFilterOptions{
        .t60s = {1.8f, 1.7f, 1.6f, 1.5f, 1.4f, 1.3f, 1.2f, 1.1f, 1.f, 0.9f},
        .delay = 1200.f,
        .sample_rate = 48000.f,
        .shelf_cutoff = 8000.f,
    });

    auto optimized = sfFDN::CreateAttenuationFilterBank(ten_band_options);
    REQUIRE(dynamic_cast<sfFDN::IIRFilterBank*>(optimized.get()) != nullptr);

    auto reference = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : ten_band_options.filter_configs)
    {
        reference->AddFilter(sfFDN::CreateAttenuationFilter(config));
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
        REQUIRE_THAT(optimized_output[i], Catch::Matchers::WithinAbs(reference_output[i], 1e-5f));
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
    heterogeneous_options.filter_configs[1] =
        sfFDN::TwoBandFilterOptions{.t60s = {1.5f, 0.7f}, .delay = 1200.f, .sample_rate = 48000.f};
    auto fallback = sfFDN::CreateAttenuationFilterBank(heterogeneous_options);
    REQUIRE(dynamic_cast<sfFDN::FilterBank*>(fallback.get()) != nullptr);
}

TEST_CASE("Three-band attenuation filter bank matches channel filters")
{
    sfFDN::AttenuationFilterBankOptions options;
    options.filter_configs.emplace_back(sfFDN::ThreeBandFilterOptions{
        .t60s = {1.5f, 1.f, 0.5f},
        .delay = 1000.f,
        .freqs = {800.f, 8000.f},
        .sample_rate = 48000.f,
    });
    options.filter_configs.emplace_back(sfFDN::ThreeBandFilterOptions{
        .t60s = {1.8f, 1.2f, 0.7f},
        .delay = 1200.f,
        .freqs = {600.f, 6000.f},
        .sample_rate = 48000.f,
    });

    auto optimized = sfFDN::CreateAttenuationFilterBank(options);
    REQUIRE(dynamic_cast<sfFDN::IIRFilterBank*>(optimized.get()) != nullptr);
    auto clone = optimized->Clone();

    auto reference = std::make_unique<sfFDN::FilterBank>();
    for (const auto& config : options.filter_configs)
    {
        reference->AddFilter(sfFDN::CreateAttenuationFilter(config));
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
    double signal_energy = 0.0;
    double error_energy = 0.0;
    float max_error = 0.f;
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

        for (auto i = 0u; i < input.size(); ++i)
        {
            const double reference_sample = reference_output[i];
            const double error = static_cast<double>(optimized_output[i]) - reference_sample;
            signal_energy += reference_sample * reference_sample;
            error_energy += error * error;
            max_error = std::max(max_error, static_cast<float>(std::abs(error)));
        }
    }

    const double snr = 10.0 * std::log10(signal_energy / error_energy);
    INFO("max error: " << max_error);
    INFO("SNR: " << snr << " dB");
    REQUIRE(max_error < 3e-5f);
    REQUIRE(snr > 90.0);
}
