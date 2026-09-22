#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "audio_buffer_alias.h"
#include "sffdn/sffdn.h"

#include "allocation_counter.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
constexpr sfFDN::ModulationOptions kTestModulation{
    .frequency = 0.01f,
    .amplitude = 0.1f,
    .initial_phase = 0.f,
};

class ReferenceAllpass
{
  public:
    explicit ReferenceAllpass(uint32_t delay)
        : delay_(delay, 0.0)
    {
    }

    double Tick(double input, double gain)
    {
        const double delayed = delay_[position_];
        const double complementary_gain = std::sqrt(1.0 - (gain * gain));
        const double output = (complementary_gain * delayed) - (gain * input);
        delay_[position_] = (complementary_gain * input) + (gain * delayed);
        position_ = (position_ + 1) % delay_.size();
        return output;
    }

    [[nodiscard]] double ResidualEnergy() const
    {
        double energy = 0.0;
        for (const double sample : delay_)
        {
            energy += sample * sample;
        }
        return energy;
    }

  private:
    std::vector<double> delay_;
    size_t position_{};
};

float NaiveTick(std::vector<float>& delay, size_t& position, float input, float gain)
{
    const float delayed = delay[position];
    const float stored = input + (gain * delayed);
    const float output = delayed - (gain * stored);
    delay[position] = stored;
    position = (position + 1) % delay.size();
    return output;
}

double Energy(std::span<const float> samples)
{
    double result = 0.0;
    for (const float sample : samples)
    {
        result += static_cast<double>(sample) * sample;
    }
    return result;
}

sfFDN::TimeVaryingSchroederAllpassSectionOptions SectionOptions(bool parallel = false)
{
    return {
        .delays = {3.f, 7.f, 11.f},
        .gains = {0.35f, -0.4f, 0.25f},
        .time_varying_config =
            {
                {.frequency = 0.015625f, .amplitude = 0.1f, .initial_phase = 0.f},
                {.frequency = 0.03125f, .amplitude = 0.2f, .initial_phase = 0.125f},
                {.frequency = 0.0625f, .amplitude = -0.15f, .initial_phase = 0.375f},
            },
        .parallel = parallel,
    };
}

void Process(sfFDN::AudioProcessor& processor, std::span<float> input, std::span<float> output)
{
    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer output_buffer(output);
    processor.Process(input_buffer, output_buffer);
}
} // namespace

TEST_CASE("TimeVaryingSchroederAllpass explicit gain matches SchroederAllpass", "[time_varying_allpass]")
{
    constexpr uint32_t kDelay = 13;
    constexpr float kGain = -0.73f;
    constexpr uint32_t kSamples = 257;

    sfFDN::TimeVaryingSchroederAllpass time_varying(kDelay, kGain, kTestModulation);
    sfFDN::SchroederAllpass static_allpass(kDelay, kGain);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        const float input = distribution(rng);
        REQUIRE_THAT(time_varying.Tick(input, kGain), Catch::Matchers::WithinAbs(static_allpass.Tick(input), 2.e-6f));
    }
}

TEST_CASE("TimeVaryingSchroederAllpass follows normalized Type V reference", "[time_varying_allpass]")
{
    constexpr uint32_t kDelay = 9;
    constexpr uint32_t kSamples = 401;
    std::array<std::vector<float>, 4> gains;
    for (auto& trajectory : gains)
    {
        trajectory.resize(kSamples);
    }

    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        gains[0][sample] = (sample < 91) ? -0.8f : 0.65f;
        gains[1][sample] = 0.82f * std::sin(0.071f * static_cast<float>(sample));
        gains[2][sample] = (sample % 3 == 0) ? 0.f : ((sample % 3 == 1) ? 0.995f : -0.995f);
    }
    std::mt19937 rng(8675309);
    std::uniform_real_distribution<float> gain_distribution(-0.97f, 0.97f);
    for (float& gain : gains[3])
    {
        gain = gain_distribution(rng);
    }

    std::mt19937 input_rng(42);
    std::uniform_real_distribution<float> input_distribution(-1.f, 1.f);
    std::vector<float> input(kSamples);
    for (float& sample : input)
    {
        sample = input_distribution(input_rng);
    }

    for (const auto& trajectory : gains)
    {
        sfFDN::TimeVaryingSchroederAllpass allpass(kDelay, 0.f, kTestModulation);
        ReferenceAllpass reference(kDelay);
        double input_energy = 0.0;
        double output_energy = 0.0;

        for (uint32_t sample = 0; sample < kSamples; ++sample)
        {
            const double expected = reference.Tick(input[sample], trajectory[sample]);
            const float actual = allpass.Tick(input[sample], trajectory[sample]);
            REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(static_cast<float>(expected), 2.e-6f));

            input_energy += static_cast<double>(input[sample]) * input[sample];
            output_energy += static_cast<double>(actual) * actual;
            REQUIRE_THAT(output_energy + reference.ResidualEnergy(), Catch::Matchers::WithinAbs(input_energy, 2.e-5));
        }
    }
}

TEST_CASE("TimeVaryingSchroederAllpass avoids naive changing-gain energy error", "[time_varying_allpass]")
{
    constexpr uint32_t kDelay = 5;
    constexpr uint32_t kSamples = 512;
    std::vector<float> input(kSamples, 0.f);
    input[0] = 1.f;

    ReferenceAllpass normalized(kDelay);
    sfFDN::TimeVaryingSchroederAllpass production(kDelay, 0.f, kTestModulation);
    std::vector<float> naive_delay(kDelay, 0.f);
    size_t naive_position = 0;
    double production_energy = 0.0;
    double naive_energy = 0.0;
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        const float gain = (sample == 0) ? 0.2f : -0.8f;
        static_cast<void>(normalized.Tick(input[sample], gain));
        production_energy += std::pow(production.Tick(input[sample], gain), 2.0);
        naive_energy += std::pow(NaiveTick(naive_delay, naive_position, input[sample], gain), 2.0);
    }

    REQUIRE_THAT(production_energy + normalized.ResidualEnergy(), Catch::Matchers::WithinAbs(1.0, 2.e-6));
    REQUIRE(std::abs(naive_energy + Energy(naive_delay) - 1.0) > 1.e-3);
}

TEST_CASE("TimeVaryingSchroederAllpass configured modulation is block invariant", "[time_varying_allpass]")
{
    constexpr uint32_t kSamples = 127;
    const sfFDN::ModulationOptions modulation{.frequency = 0.017f, .amplitude = 0.31f, .initial_phase = 0.2f};
    std::vector<float> input(kSamples);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        input[sample] = std::sin(0.13f * static_cast<float>(sample));
    }

    sfFDN::TimeVaryingSchroederAllpass tick_allpass(17, -0.4f, modulation);
    sfFDN::TimeVaryingSchroederAllpass block_allpass(17, -0.4f, modulation);
    std::vector<float> expected(kSamples);
    std::vector<float> actual(kSamples);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        expected[sample] = tick_allpass.Tick(input[sample]);
    }
    block_allpass.ProcessBlock(input, actual);

    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        REQUIRE_THAT(actual[sample], Catch::Matchers::WithinAbs(expected[sample], 1.e-6f));
    }
}

TEST_CASE("TimeVaryingSchroederAllpass ProcessBlockAccumulate adds its response", "[time_varying_allpass]")
{
    constexpr uint32_t kSamples = 37;
    const sfFDN::ModulationOptions modulation{.frequency = 0.017f, .amplitude = 0.31f, .initial_phase = 0.2f};
    std::vector<float> input(kSamples);
    std::vector<float> expected(kSamples);
    std::vector<float> accumulated(kSamples, 0.25f);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        input[sample] = std::sin(0.13f * static_cast<float>(sample));
    }

    sfFDN::TimeVaryingSchroederAllpass reference(17, -0.4f, modulation);
    sfFDN::TimeVaryingSchroederAllpass accumulating(17, -0.4f, modulation);
    reference.ProcessBlock(input, expected);
    accumulating.ProcessBlockAccumulate(input, accumulated);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        REQUIRE_THAT(accumulated[sample], Catch::Matchers::WithinAbs(0.25f + expected[sample], 1.e-6f));
    }
}

TEST_CASE("TimeVaryingSchroederAllpass Clear restores configured modulation phase", "[time_varying_allpass]")
{
    const sfFDN::ModulationOptions modulation{.frequency = 0.037f, .amplitude = 0.31f, .initial_phase = 0.375f};
    sfFDN::TimeVaryingSchroederAllpass cleared(11, -0.4f, modulation);
    sfFDN::TimeVaryingSchroederAllpass fresh(11, -0.4f, modulation);

    for (uint32_t sample = 0; sample < 23; ++sample)
    {
        static_cast<void>(cleared.Tick(0.1f * static_cast<float>(sample)));
    }
    cleared.Clear();

    for (uint32_t sample = 0; sample < 64; ++sample)
    {
        const float input = std::sin(0.11f * static_cast<float>(sample));
        REQUIRE_THAT(cleared.Tick(input), Catch::Matchers::WithinAbs(fresh.Tick(input), 1.e-6f));
    }
}

TEST_CASE("TimeVaryingSchroederAllpassSection handles blocks, tails, aliases, clone, and Clear",
          "[time_varying_allpass]")
{
    constexpr uint32_t kSamples = 95;
    std::vector<float> input(kSamples, 0.f);
    input[0] = 1.f;
    input[31] = -0.4f;

    sfFDN::TimeVaryingSchroederAllpassSection whole(SectionOptions());
    sfFDN::TimeVaryingSchroederAllpassSection partitioned(SectionOptions());
    std::vector<float> whole_output(kSamples);
    std::vector<float> partitioned_output(kSamples);
    Process(whole, input, whole_output);
    for (uint32_t start = 0; start < kSamples;)
    {
        const uint32_t count = std::min<uint32_t>(7 + (start % 13), kSamples - start);
        Process(partitioned, std::span(input).subspan(start, count),
                std::span(partitioned_output).subspan(start, count));
        start += count;
    }

    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        REQUIRE_THAT(partitioned_output[sample], Catch::Matchers::WithinAbs(whole_output[sample], 1.e-6f));
    }
    REQUIRE(std::abs(whole_output.back()) > 1.e-7f);

    sfFDN::TimeVaryingSchroederAllpassSection reference(SectionOptions());
    sfFDN::TimeVaryingSchroederAllpassSection in_place(SectionOptions());
    std::vector<float> reference_output(kSamples);
    auto aliased = input;
    Process(reference, input, reference_output);
    sfFDN::AudioBuffer aliased_buffer(aliased);
    in_place.Process(aliased_buffer, aliased_buffer);
    for (uint32_t sample = 0; sample < kSamples; ++sample)
    {
        REQUIRE_THAT(aliased[sample], Catch::Matchers::WithinAbs(reference_output[sample], 1.e-6f));
    }

    sfFDN::TimeVaryingSchroederAllpassSection original(SectionOptions());
    std::vector<float> discarded(kSamples);
    Process(original, input, discarded);
    auto clone = original.Clone();
    std::vector<float> original_output(kSamples);
    std::vector<float> clone_output(kSamples);
    Process(original, input, original_output);
    Process(*clone, input, clone_output);
    REQUIRE(original_output == clone_output);

    original.Clear();
    sfFDN::TimeVaryingSchroederAllpassSection fresh(SectionOptions());
    std::vector<float> cleared_output(kSamples);
    std::vector<float> fresh_output(kSamples);
    Process(original, input, cleared_output);
    Process(fresh, input, fresh_output);
    REQUIRE(cleared_output == fresh_output);
}

TEST_CASE("FilterBank keeps time-varying allpass channels independent without allocation", "[time_varying_allpass]")
{
    constexpr uint32_t kSamples = 64;
    sfFDN::MultichannelProcessorOptions options{.channels = {SectionOptions(), SectionOptions(true), SectionOptions()}};
    auto bank = std::make_unique<sfFDN::FilterBank>(options);
    std::vector<float> bank_input(3 * kSamples, 0.f);
    std::vector<float> bank_output(3 * kSamples, 0.f);
    bank_input[0] = 1.f;
    bank_input[kSamples] = -1.f;
    bank_input[(2 * kSamples) + 5U] = 0.5f;
    sfFDN::AudioBuffer const bank_input_buffer(kSamples, 3, bank_input);
    sfFDN::AudioBuffer bank_output_buffer(kSamples, 3, bank_output);

    std::array<sfFDN::TimeVaryingSchroederAllpassSection, 3> references = {
        sfFDN::TimeVaryingSchroederAllpassSection(SectionOptions()),
        sfFDN::TimeVaryingSchroederAllpassSection(SectionOptions(true)),
        sfFDN::TimeVaryingSchroederAllpassSection(SectionOptions()),
    };
    std::vector<float> expected(3 * kSamples, 0.f);
    for (uint32_t channel = 0; channel < 3U; ++channel)
    {
        Process(references[channel], std::span(bank_input).subspan(channel * kSamples, kSamples),
                std::span(expected).subspan(channel * kSamples, kSamples));
    }

    bank->Process(bank_input_buffer, bank_output_buffer);
    for (uint32_t sample = 0; sample < (3U * kSamples); ++sample)
    {
        REQUIRE_THAT(bank_output[sample], Catch::Matchers::WithinAbs(expected[sample], 1.e-6f));
    }
    REQUIRE(std::abs(bank_output[kSamples - 1U]) > 1.e-7f);
    REQUIRE(std::abs(bank_output[(2U * kSamples) + kSamples - 1U]) > 1.e-7f);

    bank->Clear();
    std::ranges::fill(bank_output, 0.f);
    bank->Process(bank_input_buffer, bank_output_buffer);
    {
        const sfFDNTest::ScopedAllocationCounter counter;
        bank->Process(bank_input_buffer, bank_output_buffer);
        REQUIRE(counter.Count() == 0);
    }
}

TEST_CASE("TimeVaryingSchroederAllpassSection matches exact and disjoint strided views", "[time_varying_allpass]")
{
    constexpr uint32_t kStride = 24;
    constexpr uint32_t kBlockSize = 8;
    constexpr uint32_t kInputOffset = 3;
    constexpr uint32_t kOutputOffset = 14;
    constexpr std::array<float, kBlockSize> kFirstBlock = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -8.f};
    constexpr std::array<float, kBlockSize> kSecondBlock = {-0.5f, 1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f, 7.5f};

    sfFDN::TimeVaryingSchroederAllpassSection reference(SectionOptions(true));
    sfFDN::TimeVaryingSchroederAllpassSection exact_alias_processor(SectionOptions(true));
    sfFDN::TimeVaryingSchroederAllpassSection disjoint_processor(SectionOptions(true));
    std::array<float, kBlockSize> reference_output{};
    std::vector<float> exact_parent(kStride, -100.f);
    std::vector<float> disjoint_parent(kStride, -200.f);
    sfFDN::AudioBuffer const exact_parent_buffer(exact_parent);
    sfFDN::AudioBuffer const disjoint_parent_buffer(disjoint_parent);
    sfFDN::AudioBuffer exact_input = exact_parent_buffer.Offset(kInputOffset, kBlockSize);
    sfFDN::AudioBuffer exact_output = exact_parent_buffer.Offset(kInputOffset, kBlockSize);
    sfFDN::AudioBuffer disjoint_input = disjoint_parent_buffer.Offset(kInputOffset, kBlockSize);
    sfFDN::AudioBuffer disjoint_output = disjoint_parent_buffer.Offset(kOutputOffset, kBlockSize);

    for (const auto block : {kFirstBlock, kSecondBlock})
    {
        auto reference_samples = block;
        const sfFDN::AudioBuffer reference_input(reference_samples);
        sfFDN::AudioBuffer reference_buffer(reference_output);
        reference.Process(reference_input, reference_buffer);

        std::ranges::copy(block, exact_input.GetChannelSpan(0).begin());
        exact_alias_processor.Process(exact_input, exact_output);
        std::ranges::copy(block, disjoint_input.GetChannelSpan(0).begin());
        const auto disjoint_parent_before = disjoint_parent;
        disjoint_processor.Process(disjoint_input, disjoint_output);

        for (uint32_t sample = 0; sample < kBlockSize; ++sample)
        {
            REQUIRE_THAT(exact_output.GetChannelSpan(0)[sample],
                         Catch::Matchers::WithinAbs(reference_output[sample], 1e-6f));
            REQUIRE_THAT(disjoint_output.GetChannelSpan(0)[sample],
                         Catch::Matchers::WithinAbs(reference_output[sample], 1e-6f));
        }
        REQUIRE(sfFDN::ClassifyAudioBufferAlias(disjoint_input, disjoint_output) == sfFDN::AudioBufferAlias::Disjoint);
        for (uint32_t sample = 0; sample < kBlockSize; ++sample)
        {
            REQUIRE(disjoint_input.GetChannelSpan(0)[sample] == block[sample]);
        }
        for (uint32_t sample = 0; sample < kStride; ++sample)
        {
            const bool is_output_sample = sample >= kOutputOffset && sample < kOutputOffset + kBlockSize;
            const bool is_input_sample = sample >= kInputOffset && sample < kInputOffset + kBlockSize;
            if (!is_output_sample)
            {
                REQUIRE(disjoint_parent[sample] == disjoint_parent_before[sample]);
            }
            if (!is_output_sample && !is_input_sample)
            {
                REQUIRE(disjoint_parent[sample] == -200.f);
            }
        }
    }
}

TEST_CASE("TimeVaryingSchroederAllpass validates setup options", "[time_varying_allpass]")
{
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpass(0, 0.f, kTestModulation), std::invalid_argument);
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpass(4, 1.f, kTestModulation), std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingSchroederAllpass(4, 0.7f, {.frequency = 0.1f, .amplitude = 0.3f, .initial_phase = 0.f}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingSchroederAllpass(4, 0.f, {.frequency = -0.1f, .amplitude = 0.2f, .initial_phase = 0.f}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingSchroederAllpass(4, 0.f, {.frequency = 0.f, .amplitude = 0.2f, .initial_phase = 0.f}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingSchroederAllpass(4, 0.f, {.frequency = 0.1f, .amplitude = 0.f, .initial_phase = 0.f}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sfFDN::TimeVaryingSchroederAllpass(4, 0.f, {.frequency = 0.1f, .amplitude = 0.2f, .initial_phase = 1.1f}),
        std::invalid_argument);

    auto options = SectionOptions();
    options.gains.pop_back();
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(options), std::invalid_argument);

    options = SectionOptions();
    options.time_varying_config.pop_back();
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(options), std::invalid_argument);

    options = SectionOptions();
    options.time_varying_config.clear();
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(options), std::invalid_argument);

    options = SectionOptions();
    options.delays[0] = 1.5f;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(options), std::invalid_argument);

    options = SectionOptions();
    options.delays[0] = 0.f;
    REQUIRE_THROWS_AS(sfFDN::TimeVaryingSchroederAllpassSection(options), std::invalid_argument);
}
