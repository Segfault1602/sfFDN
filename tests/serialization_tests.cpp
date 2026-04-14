#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "rng.h"
#include "sffdn/sffdn.h"
#include "test_utils.h"

namespace
{
constexpr uint32_t kBlockSize = 128;
}

template <typename T1, typename T2>
void TestAudioProcessor(T1* original, T2* deserialized)
{
    REQUIRE(deserialized->InputChannelCount() == original->InputChannelCount());
    REQUIRE(deserialized->OutputChannelCount() == original->OutputChannelCount());

    // Test that the JSON representations are the same
    nlohmann::json original_json = original->ToJson();
    nlohmann::json deserialized_json = deserialized->ToJson();
    REQUIRE(original_json == deserialized_json);

    constexpr uint32_t kBlockCount = 100;
    constexpr uint32_t sample_count = kBlockSize * kBlockCount;

    const uint32_t input_buffer_size = sample_count * original->InputChannelCount();
    const uint32_t output_buffer_size = sample_count * original->OutputChannelCount();

    std::vector<float> input_buffer(input_buffer_size, 0.f);
    std::vector<float> output_buffer(output_buffer_size, 0.f);
    std::vector<float> deserialized_output_buffer(output_buffer_size, 0.f);

    sfFDN::RNG rng;
    for (auto& sample : input_buffer)
    {
        sample = rng();
    }

    sfFDN::AudioBuffer original_input(sample_count, original->InputChannelCount(), input_buffer);
    sfFDN::AudioBuffer original_output(sample_count, original->OutputChannelCount(), output_buffer);
    sfFDN::AudioBuffer deserialized_output(sample_count, deserialized->OutputChannelCount(),
                                           deserialized_output_buffer);

    for (auto i = 0u; i < kBlockCount; ++i)
    {
        const sfFDN::AudioBuffer input_block = original_input.Offset(i * kBlockSize, kBlockSize);
        sfFDN::AudioBuffer original_output_block = original_output.Offset(i * kBlockSize, kBlockSize);
        sfFDN::AudioBuffer deserialized_output_block = deserialized_output.Offset(i * kBlockSize, kBlockSize);

        original->Process(input_block, original_output_block);
        deserialized->Process(input_block, deserialized_output_block);
    }

    for (auto i = 0u; i < output_buffer_size; ++i)
    {
        REQUIRE_THAT(deserialized_output_buffer[i], Catch::Matchers::WithinAbs(output_buffer[i], 1e-6));
    }
}

TEST_CASE("Json_FDN", "[serialization]")
{
    constexpr uint32_t kFDNOrder = 8;
    auto fdn = CreateFDN(kBlockSize, kFDNOrder);

    nlohmann::json j = fdn->ToJson();
    auto deserialized_fdn = sfFDN::FDN::FromJson(j);
    TestAudioProcessor(fdn.get(), &deserialized_fdn);
}

TEST_CASE("Json_DelayBank", "[serialization]")
{
    std::vector<float> delays = {4, 7, 13, 23, 37, 61, 97, 151};
    sfFDN::DelayBank delay_bank({delays, kBlockSize});

    nlohmann::json j = delay_bank.ToJson();
    auto deserialized_delay_bank = sfFDN::DelayBank::FromJson(j);

    TestAudioProcessor(&delay_bank, deserialized_delay_bank.get());
}

TEST_CASE("Json_FilterBank", "[serialization]")
{
    constexpr uint32_t kChannelCount = 4;
    constexpr uint32_t kFilterOrder = 11;
    auto filter_bank = GetFilterBank(kChannelCount, kFilterOrder);

    nlohmann::json j = filter_bank->ToJson();
    auto deserialized_filter_bank = sfFDN::FilterBank::FromJson(j);
    TestAudioProcessor(filter_bank.get(), deserialized_filter_bank.get());
}

TEST_CASE("Json_FilterFeedbackMatrix", "[serialization]")
{
    constexpr uint32_t kChannelCount = 4;
    auto filter_feedback_matrix = CreateFFM(kChannelCount, 4, 1.5);

    nlohmann::json j = filter_feedback_matrix->ToJson();
    auto deserialized_filter_feedback_matrix = sfFDN::FilterFeedbackMatrix::FromJson(j);
    TestAudioProcessor(filter_feedback_matrix.get(), deserialized_filter_feedback_matrix.get());
}

TEST_CASE("Json_ParallelGains", "[serialization]")
{
    constexpr std::array<float, 4> gains = {0.5f, 1.0f, 1.5f, 2.0f};
    {
        auto parallel_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge, gains);

        nlohmann::json j = parallel_gains->ToJson();
        auto deserialized_parallel_gains = sfFDN::ParallelGains::FromJson(j);
        TestAudioProcessor(parallel_gains.get(), deserialized_parallel_gains.get());
    }

    {
        auto parallel_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split, gains);

        nlohmann::json j = parallel_gains->ToJson();
        auto deserialized_parallel_gains = sfFDN::ParallelGains::FromJson(j);
        TestAudioProcessor(parallel_gains.get(), deserialized_parallel_gains.get());
    }

    {
        auto parallel_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel, gains);

        nlohmann::json j = parallel_gains->ToJson();
        auto deserialized_parallel_gains = sfFDN::ParallelGains::FromJson(j);
        TestAudioProcessor(parallel_gains.get(), deserialized_parallel_gains.get());
    }
}

TEST_CASE("Json_CascadedBiquads", "[serialization]")
{
    auto tc_filter = GetDefaultTCFilter();

    nlohmann::json j = tc_filter->ToJson();
    auto deserialized_tc_filter = sfFDN::CascadedBiquads::FromJson(j);
    TestAudioProcessor(tc_filter.get(), deserialized_tc_filter.get());
}

TEST_CASE("FDNConfig2")
{
    sfFDN::FDNConfig2 config;
    config.fdn_size = 4;
    config.transposed = false;
    config.direct_gain = 0.5f;
    config.block_size = 128;
    config.sample_rate = 48000;
    config.delay_bank_config = {
        {4, 7, 13, 23},
        128,
        sfFDN::DelayInterpolationType::None,
    };

    config.input_block_config.single_channel_processors = {sfFDN::AllpassFilterConfig{.coeff = 0.5f},
                                                           sfFDN::DelayConfig{.delay = 64}};
    config.input_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Split, {0.5f, 0.3f, 0.4f, 0.8f}, {}};

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixConfig{
        .matrix_size = 4,
        .type = sfFDN::ScalarMatrixType::Hadamard,
    };

    sfFDN::AttenuationFilterBankConfig attenuation_filter_bank_config;
    for (size_t i = 0; i < 4; ++i)
    {
        attenuation_filter_bank_config.filter_configs.push_back(sfFDN::TwoBandFilterConfig{
            .t60s = {1.f, 0.5f},
            .delay = 64.f,
            .sample_rate = 48000.f,
        });
    }
    config.loop_filter_configs.push_back(attenuation_filter_bank_config);

    config.output_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Merge, {0.7f, 0.6f, 0.5f, 0.4f}, {}};

    nlohmann::json j = config;

    std::cout << j.dump(4) << std::endl;

    sfFDN::FDNConfig2 deserialized_config = j.get<sfFDN::FDNConfig2>();
}