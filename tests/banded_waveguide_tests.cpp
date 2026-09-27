#include "allocation_counter.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sffdn/sffdn.h"

namespace
{

// The round-trip gain is frequency independent, so scaling the numerator applies it without a separate processor.
sfFDN::FilterCoefficients MakeBandedWaveguideFilter(float frequency, float sample_rate, float loop_gain)
{
    constexpr float kBandwidthHz = 32.F;
    const float radius = 1.F - std::numbers::pi_v<float> * kBandwidthHz / sample_rate;
    const float omega = 2.F * std::numbers::pi_v<float> * frequency / sample_rate;
    const float numerator = loop_gain * (1.F - radius * radius) * 0.5F;

    return {
        .b0 = numerator,
        .b1 = 0.F,
        .b2 = -numerator,
        .a0 = 1.F,
        .a1 = -2.F * radius * std::cos(omega),
        .a2 = radius * radius,
    };
}

float Energy(std::span<const float> samples)
{
    float energy = 0.F;
    for (const float sample : samples)
    {
        energy += sample * sample;
    }
    return energy;
}

constexpr float kBarSampleRate = 48000.F;
constexpr uint32_t kBarBlockSize = 1U;
constexpr uint32_t kBarModeCount = 4U;
constexpr float kBarFundamental = 220.F;
constexpr std::array kBarModeRatios = {1.F, 2.756F, 5.404F, 8.933F};
constexpr std::array kBarExcitation = {1.F, 0.85F, 0.65F, 0.45F};
constexpr std::array kBarModeT60s = {2.4F, 1.8F, 1.2F, 0.8F};

std::vector<float> RenderPluckedBar(const sfFDN::MatrixGeneratorOptions& generator, uint32_t duration_samples)
{
    sfFDN::FDNConfig config;
    config.fdn_size = kBarModeCount;
    config.block_size = kBarBlockSize;
    config.sample_rate = kBarSampleRate;
    config.delay_bank_config = {
        .delays = {},
        .block_size = kBarBlockSize,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.delay_bank_config.delays.reserve(kBarModeCount);

    sfFDN::MultichannelProcessorOptions loop_filters;
    loop_filters.channels.reserve(kBarModeCount);
    for (uint32_t mode = 0; mode < kBarModeCount; ++mode)
    {
        const float mode_frequency = kBarFundamental * kBarModeRatios[mode];
        const float delay = kBarSampleRate / mode_frequency;
        const float loop_gain = std::pow(10.F, -3.F * delay / (kBarModeT60s[mode] * kBarSampleRate));
        config.delay_bank_config.delays.push_back(delay);
        loop_filters.channels.emplace_back(sfFDN::CascadedBiquadsOptions{
            .coeffs = {MakeBandedWaveguideFilter(mode_frequency, kBarSampleRate, loop_gain)},
        });
    }

    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kBarExcitation.begin(), kBarExcitation.end()),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = kBarModeCount,
                .generator = generator,
            },
    };
    config.loop_filter_configs.emplace_back(std::move(loop_filters));
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kBarModeCount, 1.F),
        .time_varying_config = {},
    };

    auto fdn = sfFDN::CreateFDNFromConfig(config);
    std::vector<float> input(duration_samples, 0.F);
    std::vector<float> output(duration_samples, 0.F);
    input.front() = 1.F;

    for (uint32_t offset = 0; offset < duration_samples; offset += kBarBlockSize)
    {
        const sfFDN::AudioBuffer input_buffer(kBarBlockSize, 1U, std::span(input).subspan(offset, kBarBlockSize));
        sfFDN::AudioBuffer output_buffer(kBarBlockSize, 1U, std::span(output).subspan(offset, kBarBlockSize));
        fdn->Process(input_buffer, output_buffer);
    }

    return output;
}

} // namespace

TEST_CASE("BandedWaveguide.PluckedBar_Writes_Audition", "[banded_waveguide][.diagnostic]")
{
    constexpr uint32_t kDurationSamples = 10U * static_cast<uint32_t>(kBarSampleRate);

    const std::vector<float> output = RenderPluckedBar(sfFDN::ScalarMatrixType::Identity, kDurationSamples);

    REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    REQUIRE(Energy(std::span(output).first(kDurationSamples / 2U)) >
            Energy(std::span(output).last(kDurationSamples / 2U)));

    WriteWavFile("banded_waveguide_plucked_bar.wav", output);
    REQUIRE(std::filesystem::exists("test_outputs/banded_waveguide_plucked_bar.wav"));
}

// Sweeps the banded waveguide from decoupled modes towards a fully mixing feedback matrix.
TEST_CASE("BandedWaveguide.PluckedBar_Diffusion_Sweep", "[banded_waveguide][.diagnostic]")
{
    constexpr uint32_t kDurationSamples = 4U * static_cast<uint32_t>(kBarSampleRate);
    constexpr std::array kDiffusionPercents = {0U, 5U, 10U, 25U, 50U, 75U, 100U};

    const std::vector<float> identity = RenderPluckedBar(sfFDN::ScalarMatrixType::Identity, kDurationSamples);

    for (const uint32_t percent : kDiffusionPercents)
    {
        CAPTURE(percent);
        const float diffusion = static_cast<float>(percent) / 100.F;
        const std::vector<float> output =
            RenderPluckedBar(sfFDN::VariableDiffusionOptions{.diffusion = diffusion}, kDurationSamples);

        REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        REQUIRE(Energy(std::span(output).first(kDurationSamples / 2U)) >
                Energy(std::span(output).last(kDurationSamples / 2U)));

        // Zero diffusion leaves the modes decoupled, so it must reproduce the identity feedback matrix.
        if (percent == 0U)
        {
            for (size_t sample = 0; sample < output.size(); ++sample)
            {
                REQUIRE_THAT(output[sample], Catch::Matchers::WithinAbs(identity[sample], 1e-6));
            }
        }

        const std::string filename = std::format("banded_waveguide_plucked_bar_diffusion_{:03}.wav", percent);
        WriteWavFile(filename, output);
        REQUIRE(std::filesystem::exists("test_outputs/" + filename));
    }
}

namespace
{

constexpr float kBowlSampleRate = 48000.F;
constexpr uint32_t kBowlBlockSize = 1U;
constexpr uint32_t kBowlModeCount = 12U;
constexpr float kBowlFundamental = 220.F;
constexpr std::array kBowlModeRatios = {
    0.996108344F, 1.0038916562F, 2.979178F, 2.99329767F, 5.704452F,   5.704452F,
    8.9982F,      9.01549726F,   12.83303F, 12.807382F,  17.2808219F, 21.97602739726F,
};
constexpr std::array kBowlModeGains = {
    0.999925960128219F,
    0.999925960128219F,
    0.999982774366897F,
    0.999982774366897F,
    1.F,
    1.F,
    1.F,
    1.F,
    0.999965497558225F,
    0.999965497558225F,
    1.F,
    0.999999999999999965F,
};
constexpr std::array kBowlExcitation = {
    1.1900357F, 1.1900357F, 1.0914886F, 1.0914886F, 4.2995041F, 4.2995041F,
    4.0063034F, 4.0063034F, 0.7063034F, 0.7063034F, 5.7063034F, 5.7063034F,
};

sfFDN::ScalarFeedbackMatrixOptions MakeIdentityFeedback(uint32_t matrix_size);

class BowJunction final : public sfFDN::AudioProcessor
{
  public:
    BowJunction(std::vector<float> mode_gains, float pressure)
        : mode_gains_(std::move(mode_gains))
    {
        SetPressure(pressure);
    }

    void SetBowVelocity(float velocity) noexcept
    {
        bow_velocity_ = velocity;
    }

    void SetPressure(float pressure) noexcept
    {
        const float normalized_pressure = std::clamp(pressure, 0.F, 1.F);
        slope_ = 10.F - 9.F * normalized_pressure;
    }

    void Process(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output) noexcept SFFDN_NONBLOCKING override
    {
        assert(input.ChannelCount() == mode_gains_.size());
        assert(output.ChannelCount() == mode_gains_.size());
        assert(input.SampleCount() == output.SampleCount());

        for (uint32_t sample = 0; sample < input.SampleCount(); ++sample)
        {
            float bar_velocity = 0.F;
            for (uint32_t channel = 0; channel < mode_gains_.size(); ++channel)
            {
                bar_velocity += kVelocityGain * input.GetChannelSpan(channel)[sample];
            }

            const float differential_velocity = bow_velocity_ - bar_velocity;
            const float table_input = std::abs(slope_ * differential_velocity) + 0.75F;
            const float squared = table_input * table_input;
            const float reflection = std::clamp(1.F / (squared * squared), 0.01F, 0.98F);
            const float bow_force = differential_velocity * reflection / static_cast<float>(mode_gains_.size());

            for (uint32_t channel = 0; channel < mode_gains_.size(); ++channel)
            {
                output.GetChannelSpan(channel)[sample] =
                    mode_gains_[channel] * input.GetChannelSpan(channel)[sample] + bow_force;
            }
        }
    }

    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return static_cast<uint32_t>(mode_gains_.size());
    }

    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return static_cast<uint32_t>(mode_gains_.size());
    }

    void Clear() override
    {
    }

    std::unique_ptr<sfFDN::AudioProcessor> Clone() const override
    {
        return std::make_unique<BowJunction>(*this);
    }

  private:
    static constexpr float kVelocityGain = 0.999F;

    std::vector<float> mode_gains_;
    float bow_velocity_ = 0.F;
    float slope_ = 3.F;
};

std::unique_ptr<sfFDN::FDN> MakeTibetanPrayerBowl(sfFDN::ScalarFeedbackMatrixOptions feedback_matrix,
                                                  bool fold_mode_gains_into_filters)
{
    sfFDN::FDNConfig config;
    config.fdn_size = kBowlModeCount;
    config.block_size = kBowlBlockSize;
    config.sample_rate = kBowlSampleRate;
    config.delay_bank_config = {
        .delays = {},
        .block_size = kBowlBlockSize,
        .interpolation_type = sfFDN::DelayInterpolationType::None,
    };
    config.delay_bank_config.delays.reserve(kBowlModeCount);

    sfFDN::MultichannelProcessorOptions loop_filters;
    loop_filters.channels.reserve(kBowlModeCount);
    for (uint32_t mode = 0; mode < kBowlModeCount; ++mode)
    {
        const float mode_frequency = kBowlFundamental * kBowlModeRatios[mode];
        const float delay = std::floor(kBowlSampleRate / mode_frequency);
        const float filter_gain = fold_mode_gains_into_filters ? kBowlModeGains[mode] : 1.F;
        config.delay_bank_config.delays.push_back(delay);
        loop_filters.channels.emplace_back(sfFDN::CascadedBiquadsOptions{
            .coeffs = {MakeBandedWaveguideFilter(mode_frequency, kBowlSampleRate, filter_gain)},
        });
    }

    config.input_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kBowlExcitation.begin(), kBowlExcitation.end()),
        .time_varying_config = {},
    };
    config.feedback_matrix_config = std::move(feedback_matrix);
    config.loop_filter_configs.emplace_back(std::move(loop_filters));
    config.output_block_config.parallel_gains_config = {
        .gains = std::vector<float>(kBowlModeCount, 4.F),
        .time_varying_config = {},
    };

    return sfFDN::CreateFDNFromConfig(config);
}

std::vector<float> RenderTibetanPrayerBowl(sfFDN::ScalarFeedbackMatrixOptions feedback_matrix,
                                           uint32_t duration_samples)
{
    auto fdn = MakeTibetanPrayerBowl(std::move(feedback_matrix), true);
    std::vector<float> input(duration_samples, 0.F);
    std::vector<float> output(duration_samples, 0.F);
    input.front() = 1.F;

    for (uint32_t offset = 0; offset < duration_samples; offset += kBowlBlockSize)
    {
        const sfFDN::AudioBuffer input_buffer(kBowlBlockSize, 1U, std::span(input).subspan(offset, kBowlBlockSize));
        sfFDN::AudioBuffer output_buffer(kBowlBlockSize, 1U, std::span(output).subspan(offset, kBowlBlockSize));
        fdn->Process(input_buffer, output_buffer);
    }

    return output;
}

std::vector<float> RenderBowedTibetanPrayerBowl(float pressure, uint32_t duration_samples)
{
    auto fdn = MakeTibetanPrayerBowl(MakeIdentityFeedback(kBowlModeCount), false);
    auto bow_junction =
        std::make_unique<BowJunction>(std::vector<float>(kBowlModeGains.begin(), kBowlModeGains.end()), pressure);
    BowJunction* bow_control = bow_junction.get();

    auto loop_filter = std::make_unique<sfFDN::AudioProcessorChain>(kBowlBlockSize);
    const bool bow_added = loop_filter->AddProcessor(std::move(bow_junction));
    const bool filters_added = loop_filter->AddProcessor(fdn->GetLoopFilter()->Clone());
    if (!bow_added || !filters_added || !fdn->SetLoopFilter(std::move(loop_filter)))
    {
        throw std::runtime_error("Unable to install the bowed banded-waveguide loop filter");
    }

    std::vector<float> input(duration_samples, 0.F);
    std::vector<float> output(duration_samples, 0.F);
    constexpr float kMaximumVelocity = 0.08F;
    const uint32_t attack_samples = static_cast<uint32_t>(0.05F * kBowlSampleRate);
    const uint32_t release_samples = static_cast<uint32_t>(0.75F * kBowlSampleRate);
    const uint32_t release_start = duration_samples - release_samples;

    for (uint32_t sample = 0; sample < duration_samples; ++sample)
    {
        float envelope = 1.F;
        if (sample < attack_samples)
        {
            envelope = static_cast<float>(sample) / static_cast<float>(attack_samples);
        }
        else if (sample >= release_start)
        {
            envelope = static_cast<float>(duration_samples - sample) / static_cast<float>(release_samples);
        }
        bow_control->SetBowVelocity(kMaximumVelocity * envelope);

        const sfFDN::AudioBuffer input_buffer(kBowlBlockSize, 1U, std::span(input).subspan(sample, kBowlBlockSize));
        sfFDN::AudioBuffer output_buffer(kBowlBlockSize, 1U, std::span(output).subspan(sample, kBowlBlockSize));
        fdn->Process(input_buffer, output_buffer);
    }

    return output;
}

// Block-diagonal feedback matrix that rotates each adjacent (near-unison) mode pair by the same angle. A rotation has
// eigenvalues exp(+-j*angle), so each pair splits into two resonances whose separation grows with the angle.
sfFDN::MatrixData MakePairCouplingMatrix(float angle_radians)
{
    std::vector<float> coefficients(kBowlModeCount * kBowlModeCount, 0.F);
    const float c = std::cos(angle_radians);
    const float s = std::sin(angle_radians);
    for (uint32_t pair = 0; pair < kBowlModeCount; pair += 2U)
    {
        coefficients[(pair * kBowlModeCount) + pair] = c;
        coefficients[(pair * kBowlModeCount) + pair + 1U] = s;
        coefficients[((pair + 1U) * kBowlModeCount) + pair] = -s;
        coefficients[((pair + 1U) * kBowlModeCount) + pair + 1U] = c;
    }
    return {kBowlModeCount, std::move(coefficients)};
}

sfFDN::ScalarFeedbackMatrixOptions MakeIdentityFeedback(uint32_t matrix_size)
{
    return {
        .source =
            sfFDN::GeneratedMatrixOptions{
                .matrix_size = matrix_size,
                .generator = sfFDN::ScalarMatrixType::Identity,
            },
    };
}

} // namespace

TEST_CASE("BandedWaveguide.TibetanPrayerBowl_Writes_Audition", "[banded_waveguide][.diagnostic]")
{
    constexpr uint32_t kDurationSamples = 3U * static_cast<uint32_t>(kBowlSampleRate);

    const std::vector<float> output = RenderTibetanPrayerBowl(MakeIdentityFeedback(kBowlModeCount), kDurationSamples);

    REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
    REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
    REQUIRE(Energy(std::span(output).first(kDurationSamples / 2U)) >
            Energy(std::span(output).last(kDurationSamples / 2U)));

    WriteWavFile("banded_waveguide_tibetan_prayer_bowl.wav", output);
    REQUIRE(std::filesystem::exists("test_outputs/banded_waveguide_tibetan_prayer_bowl.wav"));
}

// Couples only the near-unison mode pairs, which changes their beating instead of adding damping.
TEST_CASE("BandedWaveguide.TibetanPrayerBowl_Pair_Coupling", "[banded_waveguide][.diagnostic]")
{
    constexpr uint32_t kDurationSamples = 6U * static_cast<uint32_t>(kBowlSampleRate);
    constexpr std::array kCouplingDegrees = {0U, 1U, 2U, 4U, 8U, 16U};

    const std::vector<float> uncoupled =
        RenderTibetanPrayerBowl(MakeIdentityFeedback(kBowlModeCount), kDurationSamples);

    for (const uint32_t degrees : kCouplingDegrees)
    {
        CAPTURE(degrees);
        const float angle = static_cast<float>(degrees) * std::numbers::pi_v<float> / 180.F;
        const std::vector<float> output =
            RenderTibetanPrayerBowl({.source = MakePairCouplingMatrix(angle)}, kDurationSamples);

        REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));

        if (degrees == 0U)
        {
            for (size_t sample = 0; sample < output.size(); ++sample)
            {
                REQUIRE_THAT(output[sample], Catch::Matchers::WithinAbs(uncoupled[sample], 1e-6));
            }
        }

        const std::string filename = std::format("banded_waveguide_tibetan_prayer_bowl_coupling_{:02}deg.wav", degrees);
        WriteWavFile(filename, output);
        REQUIRE(std::filesystem::exists("test_outputs/" + filename));
    }
}

TEST_CASE("BandedWaveguide.TibetanPrayerBowl_Bow_Pressure", "[banded_waveguide][.diagnostic]")
{
    constexpr uint32_t kDurationSamples = 6U * static_cast<uint32_t>(kBowlSampleRate);
    constexpr std::array kPressurePercents = {25U, 50U, 75U, 100U};

    BowJunction silent_junction(std::vector<float>(kBowlModeCount, 1.F), 0.5F);
    std::vector<float> silence(kBowlModeCount * 16U, 0.F);
    const sfFDN::AudioBuffer silent_input(16U, kBowlModeCount, silence);
    sfFDN::AudioBuffer silent_output(16U, kBowlModeCount, silence);
    {
        const sfFDNTest::ScopedAllocationCounter allocation_counter;
        silent_junction.Process(silent_input, silent_output);
        REQUIRE(allocation_counter.Count() == 0U);
    }
    REQUIRE(std::ranges::all_of(silence, [](float sample) { return sample == 0.F; }));

    for (const uint32_t percent : kPressurePercents)
    {
        CAPTURE(percent);
        const float pressure = static_cast<float>(percent) / 100.F;
        const std::vector<float> output = RenderBowedTibetanPrayerBowl(pressure, kDurationSamples);

        REQUIRE(std::ranges::any_of(output, [](float sample) { return sample != 0.F; }));
        REQUIRE(std::ranges::all_of(output, [](float sample) { return std::isfinite(sample); }));
        const float peak = std::abs(*std::ranges::max_element(
            output, [](float left, float right) { return std::abs(left) < std::abs(right); }));
        REQUIRE(peak < 10.F);
        if (percent <= 50U)
        {
            const auto sustained = std::span(output).subspan(3U * static_cast<uint32_t>(kBowlSampleRate),
                                                             static_cast<uint32_t>(kBowlSampleRate));
            REQUIRE(Energy(sustained) / static_cast<float>(sustained.size()) > 1e-4F);
        }

        const std::string filename =
            std::format("banded_waveguide_tibetan_prayer_bowl_bowed_pressure_{:03}.wav", percent);
        WriteWavFile(filename, output);
        REQUIRE(std::filesystem::exists("test_outputs/" + filename));
    }
}
