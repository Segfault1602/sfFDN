// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"
#include "sffdn/oscillator.h"
#include "sffdn/types.h"

#include <span>
#include <variant>

namespace sfFDN
{

/** @brief A delay line with time-varying delay.
 * @ingroup AudioProcessors
 */
class DelayTimeVarying : public AudioProcessor
{
  public:
    /** @brief Constructs a delay line with time-varying delay.
     * @param config The configuration options for the delay line.
     */
    DelayTimeVarying(const DelayOptions& config);

    /** @brief Clears the delay line.*/
    void Clear() override;

    /** @brief Sets the maximum delay for the delay line.
     * @param delay The maximum delay in samples.
     */
    void SetMaximumDelay(uint32_t delay);

    /** @brief Sets the delay for the delay line.
     * @param delay The delay in samples.
     */
    void SetDelay(float delay);

    /** @brief Gets the current delay of the delay line.
     * @return The delay in samples.
     */
    float GetDelay() const;

    /** @brief Sets the modulation options for the delay line.
     * @param options The modulation options.
     */
    void SetMod(const ModulationOptions& options);

    /** @brief Processes a single sample.
     * @param input The input sample.
     * @return The output sample.
     */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /**
     * @brief Returns the number of input channels this processor expects.
     * @return The number of input channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /**
     * @brief Returns the number of output channels this processor produces.
     * @return The number of output channels.
     * @note This is equal to the number of delay lines in the bank.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Creates a copy of the processor.
     * @return A unique pointer to the cloned processor.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    void UpdateDelay() noexcept SFFDN_NONBLOCKING;
    DelayInterp delay_;
    float base_delay_;

    SineWave lfo_;
};

} // namespace sfFDN