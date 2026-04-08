#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"

template <typename T>
void TestAudioProcessor(T& original, T& deserialized)
{
    REQUIRE(deserialized.InputChannelCount() == original.InputChannelCount());
    REQUIRE(deserialized.OutputChannelCount() == original.OutputChannelCount());

    // Test that the JSON representations are the same
    nlohmann::json original_json = original.ToJson();
    nlohmann::json deserialized_json = deserialized.ToJson();
    REQUIRE(original_json == deserialized_json);

    constexpr uint32_t sample_count = 32;
    const uint32_t input_buffer_size = sample_count * original.InputChannelCount();
    const uint32_t output_buffer_size = sample_count * original.OutputChannelCount();

    std::vector<float> input_buffer(input_buffer_size, 0.f);
    std::vector<float> output_buffer(output_buffer_size, 0.f);
    std::vector<float> deserialized_output_buffer(output_buffer_size, 0.f);

    sfFDN::RNG rng;
    for (auto& sample : input_buffer)
    {
        sample = rng();
    }

    sfFDN::AudioBuffer original_input(sample_count, original.InputChannelCount(), input_buffer);
    sfFDN::AudioBuffer original_output(sample_count, original.OutputChannelCount(), output_buffer);
    sfFDN::AudioBuffer deserialized_output(sample_count, deserialized.OutputChannelCount(), deserialized_output_buffer);

    original.Process(original_input, original_output);
    deserialized.Process(original_input, deserialized_output);

    for (auto i = 0u; i < output_buffer_size; ++i)
    {
        REQUIRE_THAT(deserialized_output_buffer[i], Catch::Matchers::WithinAbs(output_buffer[i], 1e-6));
    }
}

TEST_CASE("Json_DelayBank", "[serialization]")
{
    std::vector<uint32_t> delays = {4, 7, 13, 23, 37, 61, 97, 151};
    sfFDN::DelayBank delay_bank(delays, 64);

    nlohmann::json j = delay_bank.ToJson();
    auto deserialized_delay_bank = sfFDN::DelayBank::FromJson(j);

    TestAudioProcessor(delay_bank, deserialized_delay_bank);
}