#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"
#include "sffdn/oscillator.h"

#include <span>
#include <variant>

namespace sfFDN
{

struct DelayTimeVaryingConfig
{
    float delay;
    uint32_t max_delay;
    sfFDN::DelayInterpolationType interp_type{sfFDN::DelayInterpolationType::Allpass};
    float lfo_frequencies;
    float lfo_amplitudes;
    float lfo_initial_phases;
};

class DelayTimeVarying : public AudioProcessor
{
  public:
    DelayTimeVarying(float delay = 0.5, uint32_t max_delay = 4095,
                     DelayInterpolationType type = DelayInterpolationType::Linear);

    DelayTimeVarying(const DelayTimeVaryingConfig& config);

    void Clear() override;

    void SetMaximumDelay(uint32_t delay);

    void SetDelay(float delay);

    float GetDelay() const;

    void SetMod(float freq, float amplitude, float phase_offset = 0.0f);

    float Tick(float input);

    /**
     * @brief Returns the number of input channels this processor expects.
     * @return The number of input channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t InputChannelCount() const override;

    /**
     * @brief Returns the number of output channels this processor produces.
     * @return The number of output channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t OutputChannelCount() const override;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    std::unique_ptr<AudioProcessor> Clone() const override;

    nlohmann::json ToJson() const override;

  private:
    void UpdateDelay();
    DelayInterp delay_;
    float base_delay_;

    SineWave lfo_;
};

} // namespace sfFDN