// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>

#include "sffdn/sffdn.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr uint32_t kOrder = 32U;
constexpr uint32_t kHalfOrder = kOrder / 2U;
constexpr uint32_t kBlockSize = 128U;
constexpr uint32_t kSampleRate = 48000U;
constexpr uint32_t kDurationSamples = 12U * kSampleRate;
constexpr float kBoundaryGain = 1.0F / 4.0F;
constexpr float kT60Seconds = 10.0F;

// artificial-audio/pyFDN d368364896f9987b2e30b43bcbc3dfd40dbd7e27,
// examples/example_kronecker_matrix.py. Both stereo halves deliberately use the same seeded 16-delay vector. That
// symmetry produces the slow left/right energy exchange visible in the paper's pi/80 IACC curve.
constexpr std::array<float, kOrder> kDelays = {
    1361.0F, 1381.0F, 1487.0F, 1567.0F, 1711.0F, 1753.0F, 1867.0F, 1879.0F, 2053.0F, 2179.0F, 2263.0F,
    2369.0F, 2579.0F, 3083.0F, 4505.0F, 7392.0F, 1361.0F, 1381.0F, 1487.0F, 1567.0F, 1711.0F, 1753.0F,
    1867.0F, 1879.0F, 2053.0F, 2179.0F, 2263.0F, 2369.0F, 2579.0F, 3083.0F, 4505.0F, 7392.0F,
};

struct CouplingCase
{
    uint32_t percent;
    float fraction;
};

constexpr std::array kCouplingCases = {
    CouplingCase{.percent = 5U, .fraction = 0.05F},   CouplingCase{.percent = 10U, .fraction = 0.10F},
    CouplingCase{.percent = 25U, .fraction = 0.25F},  CouplingCase{.percent = 50U, .fraction = 0.50F},
    CouplingCase{.percent = 100U, .fraction = 1.00F},
};

std::unique_ptr<sfFDN::FDN> MakeStereoCouplingFDN(float coupling_fraction)
{
    auto fdn = std::make_unique<sfFDN::FDN>(sfFDN::FDNTopology{
        .order = kOrder,
        .block_size = kBlockSize,
        .input_channel_count = 2U,
        .output_channel_count = 2U,
    });
    if (!fdn->SetDelays(kDelays))
    {
        throw std::runtime_error("Unable to set stereo cross-coupling delays");
    }

    std::vector<float> input_coefficients(2U * kOrder, 0.0F);
    for (uint32_t line = 0; line < kHalfOrder; ++line)
    {
        input_coefficients[2U * line] = kBoundaryGain;
        input_coefficients[(2U * (kHalfOrder + line)) + 1U] = kBoundaryGain;
    }
    if (!fdn->SetInputGains(std::make_unique<sfFDN::ChannelMatrix>(sfFDN::ChannelMatrixOptions{
            .input_channel_count = 2U,
            .output_channel_count = kOrder,
            .coefficients = std::move(input_coefficients),
        })))
    {
        throw std::runtime_error("Unable to set stereo cross-coupling input routing");
    }

    std::vector<float> output_coefficients(2U * kOrder, 0.0F);
    for (uint32_t line = 0; line < kHalfOrder; ++line)
    {
        output_coefficients[line] = kBoundaryGain;
        output_coefficients[kOrder + kHalfOrder + line] = kBoundaryGain;
    }
    if (!fdn->SetOutputGains(std::make_unique<sfFDN::ChannelMatrix>(sfFDN::ChannelMatrixOptions{
            .input_channel_count = kOrder,
            .output_channel_count = 2U,
            .coefficients = std::move(output_coefficients),
        })))
    {
        throw std::runtime_error("Unable to set stereo cross-coupling output routing");
    }

    std::vector<float> angles(std::bit_width(kOrder) - 1U, std::numbers::pi_v<float> / 4.0F);
    angles.back() = coupling_fraction * std::numbers::pi_v<float> / 4.0F;
    std::vector<sfFDN::KroneckerKernelType> kernels(angles.size(), sfFDN::KroneckerKernelType::Rotation);
    if (!fdn->SetFeedbackMatrix(std::make_unique<sfFDN::KroneckerFeedbackMatrix>(sfFDN::KroneckerFeedbackMatrixOptions{
            .matrix_size = kOrder,
            .angles = std::move(angles),
            .kernel_types = std::move(kernels),
        })))
    {
        throw std::runtime_error("Unable to set stereo cross-coupling feedback matrix");
    }

    const sfFDN::HomogenousFilterOptions attenuation{
        .t60 = kT60Seconds,
        .delay = 0.0F,
        .sample_rate = static_cast<float>(kSampleRate),
    };
    if (!fdn->SetLoopFilter(sfFDN::CreateAttenuationFilterBank(attenuation, kDelays)))
    {
        throw std::runtime_error("Unable to set Figure 4 attenuation filters");
    }
    fdn->SetTCFilter(nullptr);
    fdn->SetDirectGain(0.0F);
    return fdn;
}

std::vector<float> RenderStereoSignal(sfFDN::FDN& fdn, std::span<const float> left_input, uint32_t tail_samples)
{
    const uint32_t sample_count = static_cast<uint32_t>(left_input.size()) + tail_samples;
    std::vector<float> planar_input(2U * sample_count, 0.0F);
    std::ranges::copy(left_input, planar_input.begin());
    std::vector<float> planar_output(2U * sample_count, 0.0F);
    const sfFDN::AudioBuffer input_buffer(sample_count, 2U, planar_input);
    sfFDN::AudioBuffer output_buffer(sample_count, 2U, planar_output);
    fdn.Process(input_buffer, output_buffer);
    return InterleaveAudioBuffer(output_buffer);
}

std::vector<float> RenderStereoImpulseResponse(sfFDN::FDN& fdn)
{
    constexpr std::array impulse = {1.0F};
    return RenderStereoSignal(fdn, impulse, kDurationSamples - 1U);
}

std::string OutputFilename(uint32_t percent)
{
    std::string digits = std::to_string(percent);
    digits.insert(digits.begin(), 3U - digits.size(), '0');
    return "kronecker_stereo_coupling_" + digits + "_ir.wav";
}

std::string PianoOutputFilename(uint32_t percent)
{
    std::string digits = std::to_string(percent);
    digits.insert(digits.begin(), 3U - digits.size(), '0');
    return "kronecker_stereo_coupling_" + digits + "_piano.wav";
}

double WindowIACC(std::span<const float> interleaved, size_t first_frame, size_t window_frames)
{
    constexpr int32_t kMaxLag = static_cast<int32_t>(kSampleRate / 1000U);
    double left_energy = 0.0;
    double right_energy = 0.0;
    for (size_t frame = 0; frame < window_frames; ++frame)
    {
        const double left = interleaved[2U * (first_frame + frame)];
        const double right = interleaved[(2U * (first_frame + frame)) + 1U];
        left_energy += left * left;
        right_energy += right * right;
    }
    const double normalization = std::sqrt(left_energy * right_energy);
    if (normalization == 0.0)
    {
        return 0.0;
    }

    double maximum = 0.0;
    for (int32_t lag = -kMaxLag; lag <= kMaxLag; ++lag)
    {
        double correlation = 0.0;
        for (size_t left_frame = 0; left_frame < window_frames; ++left_frame)
        {
            const int64_t right_frame = static_cast<int64_t>(left_frame) + lag;
            if (right_frame >= 0 && right_frame < static_cast<int64_t>(window_frames))
            {
                correlation += static_cast<double>(interleaved[2U * (first_frame + left_frame)]) *
                               interleaved[(2U * (first_frame + static_cast<size_t>(right_frame))) + 1U];
            }
        }
        maximum = std::max(maximum, std::abs(correlation) / normalization);
    }
    return maximum;
}

void RequireLightCouplingBounce(std::span<const float> impulse_response)
{
    constexpr size_t kWindowFrames = kSampleRate / 10U;
    constexpr size_t kHopFrames = kSampleRate / 20U;
    double early_maximum = 0.0;
    double middle_minimum = 1.0;
    double rebound_maximum = 0.0;
    for (size_t start = 0; start + kWindowFrames < kDurationSamples; start += kHopFrames)
    {
        const double time = static_cast<double>(start + (kWindowFrames / 2U)) / kSampleRate;
        const double iacc = WindowIACC(impulse_response, start, kWindowFrames);
        if (time <= 2.0)
        {
            early_maximum = std::max(early_maximum, iacc);
        }
        if (time >= 2.0 && time <= 3.5)
        {
            middle_minimum = std::min(middle_minimum, iacc);
        }
        if (time >= 2.5 && time <= 5.0)
        {
            rebound_maximum = std::max(rebound_maximum, iacc);
        }
    }

    REQUIRE(early_maximum > 0.9);
    REQUIRE(middle_minimum < 0.1);
    REQUIRE(rebound_maximum > 0.5);
}

} // namespace

TEST_CASE("KroneckerFeedbackMatrix renders the stereo cross-coupling impulse responses",
          "[kronecker_feedback_matrix][.diagnostic]")
{
    // Figure 4 uses a 32-line, 2-in/2-out FDN with rotation kernels, a left-only impulse and T60 = 10 s. The 10%
    // setting is an additional companion-page example between the four couplings shown in the paper figure.
    // https://andrea-coppola-arturia.github.io/fastparametricmatrices/#group-stereo
    for (const CouplingCase coupling : kCouplingCases)
    {
        auto fdn = MakeStereoCouplingFDN(coupling.fraction);
        auto impulse_response = RenderStereoImpulseResponse(*fdn);
        REQUIRE(std::ranges::all_of(impulse_response, [](float sample) { return std::isfinite(sample); }));

        double left_energy = 0.0;
        double right_energy = 0.0;
        for (size_t frame = 0; frame < kDurationSamples; ++frame)
        {
            const double left = impulse_response[2U * frame];
            const double right = impulse_response[(2U * frame) + 1U];
            left_energy += left * left;
            right_energy += right * right;
        }
        REQUIRE(left_energy > 0.0);
        REQUIRE(right_energy > 0.0);
        if (coupling.percent == 5U)
        {
            RequireLightCouplingBounce(impulse_response);
        }

        const float peak =
            std::ranges::max(std::views::transform(impulse_response, [](float sample) { return std::abs(sample); }));
        REQUIRE(peak > 0.0F);
        const float gain = 0.95F / peak;
        for (float& sample : impulse_response)
        {
            sample *= gain;
        }
        WriteWavFile(OutputFilename(coupling.percent), impulse_response, 2U);
    }
}

TEST_CASE("KroneckerFeedbackMatrix renders the stereo cross-coupling piano auditions",
          "[kronecker_feedback_matrix][.diagnostic]")
{
    const auto piano = ReadWavFile("./tests/data/piano.wav");
    REQUIRE(piano.size() == 2U * kSampleRate);

    std::array<std::vector<float>, kCouplingCases.size()> renderings;
    float global_peak = 0.0F;
    for (size_t index = 0; index < kCouplingCases.size(); ++index)
    {
        auto fdn = MakeStereoCouplingFDN(kCouplingCases[index].fraction);
        renderings[index] = RenderStereoSignal(*fdn, piano, kDurationSamples);
        REQUIRE(std::ranges::all_of(renderings[index], [](float sample) { return std::isfinite(sample); }));
        global_peak = std::max(global_peak, std::ranges::max(std::views::transform(
                                                renderings[index], [](float sample) { return std::abs(sample); })));
    }

    REQUIRE(global_peak > 0.0F);
    const float gain = 0.95F / global_peak;
    for (size_t index = 0; index < kCouplingCases.size(); ++index)
    {
        for (float& sample : renderings[index])
        {
            sample *= gain;
        }
        WriteWavFile(PianoOutputFilename(kCouplingCases[index].percent), renderings[index], 2U);
    }
}
