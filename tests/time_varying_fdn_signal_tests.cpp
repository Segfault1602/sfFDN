#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "fft.h"
#include "sffdn/sffdn.h"

namespace
{

constexpr uint32_t kOrder = 16U;
constexpr uint32_t kSampleRate = 48000U;
constexpr uint32_t kBlockSize = 256U;
constexpr uint32_t kImpulseResponseSamples = 3U * kSampleRate;
constexpr uint32_t kLosslessSamples = 2000000U;
constexpr uint32_t kAnalysisWindow = 4096U;
constexpr uint32_t kSpectrumSize = 262144U;
constexpr float kModulationFrequency = 1.0F / static_cast<float>(kSampleRate);
constexpr std::array<float, kOrder> kDelays = {503.0F, 541.0F, 593.0F, 631.0F,  683.0F,  719.0F,  761.0F,  809.0F,
                                               857.0F, 907.0F, 953.0F, 1009.0F, 1061.0F, 1103.0F, 1151.0F, 1201.0F};

struct DecayEstimate
{
    double t60_seconds;
    double slope_db_per_second;
    uint32_t fit_samples;
};

std::vector<sfFDN::ModulationOptions> MakeModulationConfig(float amplitude)
{
    std::vector<sfFDN::ModulationOptions> config(kOrder / 2U);
    for (uint32_t rotation = 0; rotation < config.size(); ++rotation)
    {
        config[rotation] = {
            .frequency = kModulationFrequency,
            .amplitude = amplitude,
            .initial_phase = static_cast<float>((rotation * 7U) % kOrder) / static_cast<float>(kOrder),
        };
    }
    return config;
}

class LinearInterpolatingFeedbackMatrix : public sfFDN::AudioProcessor
{
  public:
    LinearInterpolatingFeedbackMatrix(uint32_t order, float lfo_frequency)
        : order_(order)
        , lfo_frequency_(lfo_frequency)
        , hadamard_(order * order, 0.0F)
    {
        const float normalization = 1.0F / std::sqrt(static_cast<float>(order_));
        for (uint32_t input_channel = 0; input_channel < order_; ++input_channel)
        {
            for (uint32_t output_channel = 0; output_channel < order_; ++output_channel)
            {
                const bool negative = (std::popcount(input_channel & output_channel) % 2) != 0;
                hadamard_[(input_channel * order_) + output_channel] = negative ? -normalization : normalization;
            }
        }
    }

    void Process(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output) noexcept SFFDN_NONBLOCKING override
    {
        for (uint32_t sample = 0; sample < input.SampleCount(); ++sample)
        {
            const float alpha = 0.5F * (1.0F + std::sin(2.0F * std::numbers::pi_v<float> * phase_));
            for (uint32_t output_channel = 0; output_channel < order_; ++output_channel)
            {
                float value = 0.0F;
                for (uint32_t input_channel = 0; input_channel < order_; ++input_channel)
                {
                    const float identity = input_channel == output_channel ? 1.0F : 0.0F;
                    const float coefficient =
                        (alpha * identity) + ((1.0F - alpha) * hadamard_[(input_channel * order_) + output_channel]);
                    value += coefficient * input.GetChannelSpan(input_channel)[sample];
                }
                output.GetChannelSpan(output_channel)[sample] = value;
            }
            phase_ += lfo_frequency_;
            phase_ -= std::floor(phase_);
        }
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return order_;
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return order_;
    }

    void Clear() override
    {
        phase_ = 0.0F;
    }

    std::unique_ptr<sfFDN::AudioProcessor> Clone() const override
    {
        return std::make_unique<LinearInterpolatingFeedbackMatrix>(*this);
    }

  private:
    uint32_t order_;
    float lfo_frequency_;
    float phase_{0.0F};
    std::vector<float> hadamard_;
};

std::unique_ptr<sfFDN::FDN> CreateFDN(float modulation_amplitude, bool add_attenuation)
{
    auto fdn = std::make_unique<sfFDN::FDN>(kOrder, kBlockSize);
    std::array<float, kOrder> gains{};
    gains.fill(1.0F / std::sqrt(static_cast<float>(kOrder)));

    if (!fdn->SetInputGains(gains) || !fdn->SetOutputGains(gains) || !fdn->SetDelays(kDelays))
    {
        throw std::runtime_error("Unable to configure time-varying FDN test fixture");
    }
    fdn->SetDirectGain(0.0F);

    auto matrix = std::make_unique<sfFDN::TimeVaryingFeedbackMatrix>(sfFDN::TimeVaryingFeedbackMatrixOptions{
        .matrix_size = kOrder, .time_varying_config = MakeModulationConfig(modulation_amplitude)});
    if (!fdn->SetFeedbackMatrix(std::move(matrix)))
    {
        throw std::runtime_error("Unable to install time-varying feedback matrix");
    }

    if (add_attenuation)
    {
        const sfFDN::HomogenousFilterOptions attenuation{
            .t60 = 1.5F, .delay = 0.0F, .sample_rate = static_cast<float>(kSampleRate)};
        if (!fdn->SetLoopFilter(sfFDN::CreateAttenuationFilterBank(attenuation, kDelays)))
        {
            throw std::runtime_error("Unable to install FDN attenuation filters");
        }
    }
    return fdn;
}

std::unique_ptr<sfFDN::FDN> CreateLinearInterpolationFDN()
{
    auto fdn = CreateFDN(0.0F, true);
    if (!fdn->SetFeedbackMatrix(std::make_unique<LinearInterpolatingFeedbackMatrix>(kOrder, kModulationFrequency)))
    {
        throw std::runtime_error("Unable to install linearly interpolating feedback matrix");
    }
    return fdn;
}

std::vector<float> RenderImpulseResponse(sfFDN::FDN& fdn, uint32_t sample_count)
{
    std::vector<float> input(sample_count, 0.0F);
    std::vector<float> output(sample_count, 0.0F);
    input.front() = 1.0F;

    sfFDN::AudioBuffer input_buffer(sample_count, 1U, input);
    sfFDN::AudioBuffer output_buffer(sample_count, 1U, output);
    fdn.Process(input_buffer, output_buffer);
    return output;
}

DecayEstimate EstimateT60(std::span<const float> impulse_response)
{
    std::vector<double> edc(impulse_response.size(), 0.0);
    double energy = 0.0;
    for (size_t index = impulse_response.size(); index-- > 0U;)
    {
        energy += static_cast<double>(impulse_response[index]) * impulse_response[index];
        edc[index] = energy;
    }

    if (energy == 0.0)
    {
        return {std::numeric_limits<double>::infinity(), 0.0, 0U};
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    uint32_t fit_samples = 0U;
    for (size_t index = 0; index < edc.size(); ++index)
    {
        const double decay_db = 10.0 * std::log10(edc[index] / energy);
        if (decay_db <= -5.0 && decay_db >= -35.0)
        {
            const double time_seconds = static_cast<double>(index) / static_cast<double>(kSampleRate);
            sum_x += time_seconds;
            sum_y += decay_db;
            sum_xx += time_seconds * time_seconds;
            sum_xy += time_seconds * decay_db;
            ++fit_samples;
        }
    }

    const double denominator = (static_cast<double>(fit_samples) * sum_xx) - (sum_x * sum_x);
    const double slope = ((static_cast<double>(fit_samples) * sum_xy) - (sum_x * sum_y)) / denominator;
    return {.t60_seconds = -60.0 / slope, .slope_db_per_second = slope, .fit_samples = fit_samples};
}

double WindowRms(std::span<const float> signal, uint32_t first_sample, uint32_t last_sample)
{
    double energy = 0.0;
    for (uint32_t sample = first_sample; sample < last_sample; ++sample)
    {
        energy += static_cast<double>(signal[sample]) * signal[sample];
    }
    return std::sqrt(energy / static_cast<double>(last_sample - first_sample));
}

float MaxAbs(std::span<const float> signal)
{
    float maximum = 0.0F;
    for (const float sample : signal)
    {
        maximum = std::max(maximum, std::abs(sample));
    }
    return maximum;
}

double EchoDensity(std::span<const float> signal, uint32_t first_sample)
{
    const auto window = signal.subspan(first_sample, kAnalysisWindow);
    double mean = 0.0;
    for (const float sample : window)
    {
        mean += sample;
    }
    mean /= static_cast<double>(window.size());

    double variance = 0.0;
    for (const float sample : window)
    {
        const double difference = static_cast<double>(sample) - mean;
        variance += difference * difference;
    }
    const double standard_deviation = std::sqrt(variance / static_cast<double>(window.size()));
    const uint32_t crossing_count =
        static_cast<uint32_t>(std::count_if(window.begin(), window.end(), [standard_deviation](float sample) {
            return std::abs(sample) > standard_deviation;
        }));
    constexpr double kGaussianCrossingRate = 0.31731050786291415;
    return (static_cast<double>(crossing_count) / static_cast<double>(window.size())) / kGaussianCrossingRate;
}

double SpectralFlatness(std::span<const float> signal)
{
    sfFDN::FFT fft;
    if (!fft.Initialize(kSpectrumSize))
    {
        throw std::runtime_error("Unable to initialize FFT for spectral-flatness analysis");
    }

    auto input = fft.AllocateRealBuffer();
    auto spectrum = fft.AllocateComplexBuffer();
    std::copy(signal.begin(), signal.end(), input.Data().begin());
    fft.Forward(input, spectrum);

    double sum_power = 0.0;
    double sum_log_power = 0.0;
    for (const auto value : spectrum.Data())
    {
        const double power = static_cast<double>(std::norm(value));
        sum_power += power;
        sum_log_power += std::log(std::max(power, 1.0e-30));
    }
    const double bin_count = static_cast<double>(spectrum.size());
    return std::exp(sum_log_power / bin_count) / (sum_power / bin_count);
}

void PrintEchoDensityReport(std::span<const float> unmodulated, std::span<const float> modulated)
{
    constexpr std::array<uint32_t, 3> kTimes = {24000U, 48000U, 96000U};
    std::cout << std::fixed << std::setprecision(5) << "Echo density (unmodulated, modulated):";
    for (const uint32_t first_sample : kTimes)
    {
        std::cout << " t=" << (static_cast<float>(first_sample) / static_cast<float>(kSampleRate)) << "s ("
                  << EchoDensity(unmodulated, first_sample) << ", " << EchoDensity(modulated, first_sample) << ")";
    }
    std::cout << '\n';
}

} // namespace

TEST_CASE("Time-varying FDN preserves T60 and reports diffusion metrics", "[time_varying_fdn]")
{
    auto unmodulated_fdn = CreateFDN(0.0F, true);
    auto modulated_fdn = CreateFDN(0.7F, true);
    const auto unmodulated = RenderImpulseResponse(*unmodulated_fdn, kImpulseResponseSamples);
    const auto modulated = RenderImpulseResponse(*modulated_fdn, kImpulseResponseSamples);

    const auto unmodulated_t60 = EstimateT60(unmodulated);
    const auto modulated_t60 = EstimateT60(modulated);
    const double t60_difference =
        std::abs(modulated_t60.t60_seconds - unmodulated_t60.t60_seconds) / unmodulated_t60.t60_seconds;
    const double unmodulated_flatness = SpectralFlatness(unmodulated);
    const double modulated_flatness = SpectralFlatness(modulated);

    INFO("unmodulated T60=" << unmodulated_t60.t60_seconds << "s, modulated T60=" << modulated_t60.t60_seconds
                            << "s, relative difference=" << t60_difference << ", fit samples=("
                            << unmodulated_t60.fit_samples << ", " << modulated_t60.fit_samples << ")");
    std::cout << std::fixed << std::setprecision(5) << "T60 (unmodulated, modulated): (" << unmodulated_t60.t60_seconds
              << " s, " << modulated_t60.t60_seconds << " s), difference=" << (100.0 * t60_difference) << "%\n";
    PrintEchoDensityReport(unmodulated, modulated);
    std::cout << "Spectral flatness (unmodulated, modulated): (" << unmodulated_flatness << ", " << modulated_flatness
              << ")\n";

    REQUIRE(unmodulated_t60.fit_samples > 1000U);
    REQUIRE(modulated_t60.fit_samples > 1000U);
    REQUIRE(std::isfinite(unmodulated_t60.t60_seconds));
    REQUIRE(std::isfinite(modulated_t60.t60_seconds));
    // A systematically contractive sin/cos approximation shortens T60 long before it becomes visibly unstable.
    REQUIRE(t60_difference < 0.05);

    // These measures are report-only: their direction is content dependent, but they should remain well-defined.
    REQUIRE(std::isfinite(unmodulated_flatness));
    REQUIRE(std::isfinite(modulated_flatness));
}

TEST_CASE("Lossless time-varying FDN remains bounded over a long run", "[time_varying_fdn]")
{
    auto fdn = CreateFDN(0.7F, false);
    const auto output = RenderImpulseResponse(*fdn, kLosslessSamples);
    const double early_rms = WindowRms(output, 200000U, 300000U);
    const double late_rms = WindowRms(output, 1900000U, 2000000U);
    const double rms_ratio = late_rms / early_rms;

    INFO("early RMS=" << early_rms << ", late RMS=" << late_rms << ", late/early=" << rms_ratio
                      << ", max abs=" << MaxAbs(output));
    std::cout << std::fixed << std::setprecision(7) << "Lossless FDN RMS (early, late, ratio): (" << early_rms << ", "
              << late_rms << ", " << rms_ratio << ")\n";

    // Per-line output energy fluctuates for unequal delays. Long output-RMS windows instead average those
    // fluctuations while exposing the loss or gain of an otherwise unitary feedback loop. A 3% tolerance admits
    // finite-window variation while readily detecting systematic contraction or expansion.
    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    REQUIRE(MaxAbs(output) < 2.0F);
    REQUIRE(rms_ratio > 0.97);
    REQUIRE(rms_ratio < 1.03);
}

TEST_CASE("Time-varying FDN does not regress to linear matrix interpolation", "[time_varying_fdn]")
{
    auto orthogonal_fdn = CreateFDN(0.7F, true);
    auto linear_fdn = CreateLinearInterpolationFDN();
    const auto orthogonal = RenderImpulseResponse(*orthogonal_fdn, kImpulseResponseSamples);
    const auto linear = RenderImpulseResponse(*linear_fdn, kImpulseResponseSamples);
    const auto orthogonal_t60 = EstimateT60(orthogonal);
    const auto linear_t60 = EstimateT60(linear);
    const double shorter_fraction = 1.0 - (linear_t60.t60_seconds / orthogonal_t60.t60_seconds);

    INFO("orthogonal T60=" << orthogonal_t60.t60_seconds << "s, linear-interpolation T60=" << linear_t60.t60_seconds
                           << "s, shorter fraction=" << shorter_fraction);
    std::cout << std::fixed << std::setprecision(5) << "T60 (orthogonal, linear interpolation): ("
              << orthogonal_t60.t60_seconds << " s, " << linear_t60.t60_seconds << " s), linear is "
              << (100.0 * shorter_fraction) << "% shorter\n";

    REQUIRE(orthogonal_t60.fit_samples > 1000U);
    REQUIRE(linear_t60.fit_samples > 1000U);
    REQUIRE(std::isfinite(orthogonal_t60.t60_seconds));
    REQUIRE(std::isfinite(linear_t60.t60_seconds));
    // This fails if TimeVaryingFeedbackMatrix is later "simplified" into a cross-fade of fixed matrices.
    REQUIRE(linear_t60.t60_seconds < (orthogonal_t60.t60_seconds * 0.95));
}
