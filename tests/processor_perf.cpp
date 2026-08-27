#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"

#include <array>
#include <chrono>
#include <memory>
#include <vector>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("AudioProcessorChainPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr std::array<sfFDN::FilterCoefficients, 2> kCoefficients = {{
        {.b0 = 0.75f, .b1 = -0.25f, .b2 = 0.1f, .a0 = 1.f, .a1 = -0.4f, .a2 = 0.2f},
        {.b0 = 0.6f, .b1 = 0.15f, .b2 = -0.05f, .a0 = 1.f, .a1 = -0.3f, .a2 = 0.1f},
    }};

    auto chain = std::make_unique<sfFDN::AudioProcessorChain>(kBlockSize);
    chain->AddProcessor(std::make_unique<sfFDN::OnePoleFilter>(0.7f, -0.3f));
    chain->AddProcessor(std::make_unique<sfFDN::AllpassFilter>(sfFDN::AllpassFilterOptions{.coeff = 0.5f}));
    auto chain_cascade = std::make_unique<sfFDN::CascadedBiquads>();
    chain_cascade->SetCoefficients(kCoefficients);
    chain->AddProcessor(std::move(chain_cascade));

    sfFDN::OnePoleFilter one_pole(0.7f, -0.3f);
    sfFDN::AllpassFilter allpass({.coeff = 0.5f});
    sfFDN::CascadedBiquads cascade;
    cascade.SetCoefficients(kCoefficients);

    std::vector<float> input(kBlockSize);
    std::vector<float> scratch_a(kBlockSize);
    std::vector<float> scratch_b(kBlockSize);
    std::vector<float> output(kBlockSize);
    sfFDN::RNG generator;
    for (float& sample : input)
    {
        sample = generator();
    }

    sfFDN::AudioBuffer input_buffer(input);
    sfFDN::AudioBuffer scratch_a_buffer(scratch_a);
    sfFDN::AudioBuffer scratch_b_buffer(scratch_b);
    sfFDN::AudioBuffer output_buffer(output);

    nanobench::Bench bench;
    bench.title("AudioProcessorChain overhead");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);
    bench.relative(true);

    bench.run("Direct concrete processors", [&] {
        one_pole.Process(input_buffer, scratch_a_buffer);
        allpass.Process(scratch_a_buffer, scratch_b_buffer);
        cascade.Process(scratch_b_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
    bench.run("AudioProcessorChain", [&] {
        chain->Process(input_buffer, output_buffer);
        nanobench::doNotOptimizeAway(output);
    });
}

TEST_CASE("DelayBankTimeVaryingPerf")
{
    constexpr uint32_t kBlockSize = 128;
    constexpr std::array<uint32_t, 2> kOrders = {8, 32};

    nanobench::Bench bench;
    bench.title("DelayBankTimeVarying scaling");
    bench.timeUnit(1us, "us");
    bench.minEpochTime(50ms);

    for (const uint32_t order : kOrders)
    {
        sfFDN::DelayBankTimeVaryingOptions options;
        options.max_delay = 8192;
        options.interpolation_type = sfFDN::DelayInterpolationType::Linear;
        options.delays.resize(order);
        options.time_varying_config.resize(order);
        for (auto channel = 0u; channel < order; ++channel)
        {
            options.delays[channel] = 1000.f + static_cast<float>(channel * 73u);
            options.time_varying_config[channel] = {
                .frequency = 0.0001f * static_cast<float>(channel + 1),
                .amplitude = 16.f,
                .initial_phase = static_cast<float>(channel) / static_cast<float>(order),
            };
        }

        sfFDN::DelayBankTimeVarying delay_bank(options);
        std::vector<float> input(kBlockSize * order);
        std::vector<float> output(kBlockSize * order);
        sfFDN::RNG generator;
        for (float& sample : input)
        {
            sample = generator();
        }
        sfFDN::AudioBuffer input_buffer(kBlockSize, order, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, order, output);

        bench.run("N=" + std::to_string(order), [&] {
            delay_bank.Process(input_buffer, output_buffer);
            nanobench::doNotOptimizeAway(output);
        });
    }
}
