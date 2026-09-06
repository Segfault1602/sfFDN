#include "sffdn/delaybank_time_varying.h"

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"

#include "processor_option_validation.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace sfFDN
{

DelayBankTimeVarying::DelayBankTimeVarying(const DelayBankTimeVaryingOptions& config)
    : config_(detail::RequireValidOptions(config))
{
    const uint32_t num_delays = config.delays.size();

    for (uint32_t i = 0; i < num_delays; i++)
    {
        DelayOptions delay_config{
            .delay = config.delays[i], .max_delay = config.max_delay, .interp_type = config.interpolation_type};
        if (!config.time_varying_config.empty())
        {
            delay_config.lfo_config = config.time_varying_config.at(i);
        }
        auto tv_delay = std::make_unique<DelayTimeVarying>(delay_config);
        delay_bank_.AddFilter(std::move(tv_delay));
    }
}

std::vector<float> DelayBankTimeVarying::GetDelays() const
{
    return config_.delays;
}

uint32_t DelayBankTimeVarying::InputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return delay_bank_.InputChannelCount();
}

uint32_t DelayBankTimeVarying::OutputChannelCount() const noexcept SFFDN_NONBLOCKING
{
    return delay_bank_.OutputChannelCount();
}

void DelayBankTimeVarying::Clear()
{
    delay_bank_.Clear();
}

void DelayBankTimeVarying::Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
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

} // namespace sfFDN