#include "sffdn/delaybank_time_varying.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sfFDN
{

DelayBankTimeVarying::DelayBankTimeVarying(const DelayBankTimeVaryingConfig& config)
    : config_(config)
{
    // validate config
    const uint32_t num_delays = config.delays.size();
    if (config.mod_freqs.size() != num_delays || config.mod_depths.size() != num_delays ||
        (!config.mod_phase_offsets.empty() && config.mod_phase_offsets.size() != num_delays))
    {
        throw std::invalid_argument(
            "DelayBankTimeVarying: size of mod_freqs, mod_depths, and mod_phase_offsets must match number of delays");
    }

    for (uint32_t i = 0; i < num_delays; i++)
    {
        auto tv_delay =
            std::make_unique<DelayTimeVarying>(config.delays[i], config.max_delay, config.interpolation_type);
        tv_delay->SetMod(config.mod_freqs[i], config.mod_depths[i],
                         config.mod_phase_offsets.empty() ? 0.0f : config.mod_phase_offsets[i]);
        delay_bank_.AddFilter(std::move(tv_delay));
    }
}

std::vector<float> DelayBankTimeVarying::GetDelays() const
{
    return config_.delays;
}

uint32_t DelayBankTimeVarying::InputChannelCount() const
{
    return delay_bank_.InputChannelCount();
}

uint32_t DelayBankTimeVarying::OutputChannelCount() const
{
    return delay_bank_.OutputChannelCount();
}

void DelayBankTimeVarying::Clear()
{
    delay_bank_.Clear();
}

void DelayBankTimeVarying::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == delay_bank_.InputChannelCount());

    delay_bank_.Process(input, output);
}

std::unique_ptr<AudioProcessor> DelayBankTimeVarying::Clone() const
{
    auto clone = std::make_unique<DelayBankTimeVarying>(config_);
    return clone;
}

nlohmann::json DelayBankTimeVarying::ToJson() const
{
    nlohmann::json j;
    j["type"] = "DelayBankTimeVarying";
    j["delay_bank_"] = delay_bank_.ToJson();
    return j;
}

} // namespace sfFDN