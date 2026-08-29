// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/delay_interp.h"
#include "sffdn/filterbank.h"
#include "sffdn/oscillator.h"
#include "sffdn/types.h"

#include <cstdint>
#include <memory>

namespace sfFDN
{

/** @brief The delay-line effect described by Jon Dattorro in "Effect Design Part 2: Delay-Line Modulation and Chorus",
 * J. Audio Eng. Soc., Vol. 45, No. 10, 1997.
 *
 * A single delay line is read at two taps and wrapped in a comb filter with three knobs: blend, feedforward and
 * feedback.
 *
 * \f[
 * w[n] = x[n] - \mathit{FB} \cdot w[n - M] \\
 * y[n] = \mathit{FF} \cdot w[n - M(n)] + \mathit{BL} \cdot w[n]
 * \f]
 *
 * The feedback tap sits at the fixed nominal delay \f$M\f$ and is read without interpolation. Only the feedforward tap
 * is modulated: it is read with interpolation at \f$M(n) = M + \mathit{width} \cdot \sin(2 \pi f n)\f$. Modulating the
 * feedback tap would change the length of the recirculating loop on every sample, so it is deliberately left fixed.
 *
 * Vibrato, flanging, chorus, doubling and echo are all obtained from this one structure by changing the three gains.
 * See MakeDattorroDelayOptions().
 *
 * @note The feedback is *subtracted* at the summing junction, as drawn in the paper. A positive feedback gain
 * therefore recirculates with inverted polarity, and the gains of Table 1 can be used exactly as printed. See
 * MakeDattorroDelayOptions().
 *
 * @note Choose the interpolation type to match how the delay is used. Linear interpolation is the cheaper choice on a
 * modulated tap and its magnitude droop near Nyquist is inaudible for an insert effect, which is what the presets
 * returned by MakeDattorroDelayOptions() use. Allpass interpolation costs a little more but has a flat magnitude
 * response; prefer it whenever the effect sits inside a feedback loop, where the droop of linear interpolation
 * compounds on every circulation and pulls the high-frequency T60 below target. Allpass interpolation is safe on a
 * modulated tap: DelayInterp::SetDelay() re-seeds the allpass state from the new tap whenever the integer part of the
 * delay changes, so crossing a sample boundary no longer leaves a step in the output.
 *
 * @ingroup AudioProcessors
 */
class DattorroDelay : public AudioProcessor
{
  public:
    /** @brief Constructs a Dattorro delay-line effect.
     * @param options The configuration options for the effect.
     * @note The instantaneous delay is never allowed to fall below kMinimumDelay samples. The constructor throws
     * std::invalid_argument if the nominal delay minus the modulation width is smaller than that.
     * @note The fixed feedback tap is placed at the nearest integer to the nominal delay, since it is read without
     * interpolation. The modulated feedforward tap keeps the fractional part of the nominal delay.
     */
    explicit DattorroDelay(const DattorroDelayOptions& options = {});

    /** @brief The smallest delay, in samples, that the delay line is allowed to take. */
    static constexpr float kMinimumDelay = 2.f;

    /** @brief Sets the gain applied to the input of the delay line. */
    void SetBlend(float blend) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the gain applied to the input of the delay line. */
    float GetBlend() const noexcept SFFDN_NONBLOCKING;

    /** @brief Sets the gain applied to the modulated output of the delay line. */
    void SetFeedforward(float feedforward) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the gain applied to the modulated output of the delay line. */
    float GetFeedforward() const noexcept SFFDN_NONBLOCKING;

    /** @brief Sets the gain applied to the fixed output of the delay line before it is fed back into the delay line.
     * @param feedback The feedback gain. The value is clamped to the range (-1, 1) to keep the loop stable.
     */
    void SetFeedback(float feedback) noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the gain applied to the fixed output of the delay line. */
    float GetFeedback() const noexcept SFFDN_NONBLOCKING;

    /** @brief Sets the nominal delay of the delay line.
     * @param delay The delay in samples.
     * @note This moves both the fixed feedback tap and the center of the modulated feedforward tap.
     * @note Throws std::invalid_argument if the delay minus the current modulation width is smaller than
     * kMinimumDelay.
     */
    void SetDelay(float delay);

    /** @brief Returns the nominal delay of the delay line, in samples. */
    float GetDelay() const;

    /** @brief Sets the modulation applied to the feedforward tap.
     * @param options The modulation options. The amplitude is the peak deviation, in samples, of the feedforward tap
     * around the nominal delay.
     * @note Throws std::invalid_argument if the nominal delay minus the amplitude is smaller than kMinimumDelay.
     */
    void SetMod(const ModulationOptions& options);

    /** @brief Processes a single sample.
     * @param input The input sample.
     * @return The output sample.
     */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * @note The input and output buffers must have one channel and the same number of samples.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels this processor expects. This is always 1. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels this processor produces. This is always 1. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the delay line and resets the modulation phase. The three gains are left untouched. */
    void Clear() override;

    /** @brief Creates a copy of the processor.
     * @return A unique pointer to the cloned processor.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float TickInternal(float input, float mod) noexcept SFFDN_NONBLOCKING;

    DelayInterp delay_;
    SineWave lfo_;

    float base_delay_;
    uint32_t feedback_tap_;

    float blend_;
    float feedforward_;
    float feedback_;
};

/** @brief Returns the options of one of the classic delay-line effects of Dattorro's Table 1.
 * @param type The effect to configure.
 * @param sample_rate The sample rate, in Hz, used to convert the delay, modulation width and modulation rate to
 * samples and to a normalized frequency.
 * @note The blend, feedforward and feedback gains are the values given in the paper. The delay, modulation width and
 * modulation rate are nominal values picked inside the ranges given in the paper.
 */
DattorroDelayOptions MakeDattorroDelayOptions(DattorroEffectType type, float sample_rate);

/** @brief Builds a bank of Dattorro delay-line effects, one per channel.
 * @param options The per-channel configurations. The number of channels of the returned processor is
 * `options.delays.size()`.
 * @return A FilterBank holding one DattorroDelay per entry of `options.delays`.
 * @note Throws std::invalid_argument if any entry has a delay smaller than DattorroDelay::kMinimumDelay, or a delay
 * whose modulation width would take the tap below it.
 */
std::unique_ptr<FilterBank> MakeMultichannelDattorroDelay(const MultichannelDattorroDelayOptions& options);

/** @brief Returns a decorrelated, multichannel version of one of the classic delay-line effects of Dattorro's Table 1.
 *
 * The gains of the effect are taken unchanged from MakeDattorroDelayOptions(), but the modulation of each channel is
 * offset from its neighbours so that the channels do not modulate coherently:
 * - the LFO initial phase of channel `i` of `N` is `i / N`;
 * - the LFO rate and the nominal delay are spread by up to +/- 10% across the channels.
 *
 * @param type The effect to configure.
 * @param sample_rate The sample rate, in Hz.
 * @param channel_count The number of channels. A count of 0 returns an empty bank, and a count of 1 returns exactly
 * the options of MakeDattorroDelayOptions().
 *
 * @note Unlike MakeDattorroDelayOptions(), the modulated presets returned here use
 * DelayInterpolationType::Allpass. The single-channel presets use linear interpolation because it is cheaper and its
 * magnitude droop near Nyquist is inaudible for an insert effect; that reasoning does not survive being placed inside
 * a feedback loop, where the droop compounds on every circulation and pulls the high-frequency T60 below target.
 * Allpass interpolation has a flat magnitude response. Echo is not modulated, so it needs no interpolation at all.
 *
 * @note A DattorroDelay with a non-zero feedback nests a recirculating loop inside whatever loop it is placed in, and
 * its worst-case gain is bounded by `(|blend| + |feedforward|) / (1 - |feedback|)`. That bound is loose while the
 * effect is static but is reached almost immediately once the tap is modulated: the pole of the comb sits at the fixed
 * feedback tap while the zero moves with the feedforward tap, so the pole-zero cancellation that makes
 * DattorroEffectType::WhiteChorus allpass only holds while the two taps coincide. Displacing the feedforward tap by
 * half a sample already takes the peak gain of the white chorus from 1.00 to 4.11 (+12.3 dB), and by two samples it
 * has saturated at 5.83 (+15.3 dB).
 *
 * The presets are therefore *not* interchangeable inside a feedback delay network:
 * - DattorroEffectType::Vibrato has no feedback and no blend, so it is a pure modulated delay with a gain of exactly
 *   1 at every frequency. It is the only modulated preset that is unconditionally safe in a feedback loop.
 * - DattorroEffectType::Doubling has no feedback either; its gain is a fixed 1.41 (+3.0 dB) that the enclosing loop
 *   has to budget for.
 * - DattorroEffectType::WhiteChorus and DattorroEffectType::Flanger both carry feedback and reach roughly +15 dB and
 *   +13 dB respectively once modulated. Placing either in an FDN feedback loop will make the network diverge.
 * - DattorroEffectType::Echo is not modulated, so it stays at its static gain of 1.18 (+1.4 dB).
 */
MultichannelDattorroDelayOptions MakeMultichannelDattorroDelayOptions(DattorroEffectType type, float sample_rate,
                                                                     uint32_t channel_count);

} // namespace sfFDN
