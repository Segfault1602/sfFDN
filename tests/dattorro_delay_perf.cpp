#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "processor_perf_utils.h"
#include "sffdn/sffdn.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace ankerl;

namespace
{
struct PresetInfo
{
    sfFDN::DattorroEffectType type;
    std::string_view name;
};

constexpr std::array kPresets = {
    PresetInfo{sfFDN::DattorroEffectType::Vibrato, "Vibrato"},
    PresetInfo{sfFDN::DattorroEffectType::Flanger, "Flanger"},
    PresetInfo{sfFDN::DattorroEffectType::WhiteChorus, "WhiteChorus"},
    PresetInfo{sfFDN::DattorroEffectType::Doubling, "Doubling"},
    PresetInfo{sfFDN::DattorroEffectType::Echo, "Echo"},
};
} // namespace

TEST_CASE("DattorroDelayPerf", "[delay]")
{
    constexpr float kSampleRate = static_cast<float>(sfFDN::kDefaultSampleRate);

    nanobench::Bench bench;
    sfFDN::test::perf::ConfigureThroughputBench(bench, "DattorroDelay perf");

    for (const PresetInfo& preset : kPresets)
    {
        for (const uint32_t block_size : sfFDN::test::perf::kBlockSizes)
        {
            std::vector<float> input(block_size);
            std::vector<float> output(block_size);
            sfFDN::test::perf::FillNoise(input);
            const sfFDN::AudioBuffer input_buffer(input);
            sfFDN::AudioBuffer output_buffer(output);
            sfFDN::DattorroDelay processor(sfFDN::MakeDattorroDelayOptions(preset.type, kSampleRate));

            sfFDN::test::perf::SetChannelSampleBatch(bench, block_size);
            bench.run(std::string(preset.name) + " B=" + std::to_string(block_size), [&] {
                processor.Process(input_buffer, output_buffer);
                nanobench::doNotOptimizeAway(output);
            });
        }
    }
}
