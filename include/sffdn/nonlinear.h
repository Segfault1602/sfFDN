// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/attributes.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filterbank.h"
#include "sffdn/oscillator.h"
#include "sffdn/types.h"

#include <cstdint>
#include <memory>

namespace sfFDN
{

class DcBlocker;

/** @defgroup ShimmerNonlinearities Shimmer nonlinearities
 * @brief Nonlinear operations meant to be placed inside the feedback loop of an FDN.
 *
 * These implement three of the five operations of G. Dal Santo, X. Pi, K. Prawda, S. J. Schlecht and V. Välimäki,
 * "Shimmer Reverberation with Nonlinear Feedback Delay Networks", Proc. DAFx26, Cambridge, MA, USA, 2026.
 *
 * Each of them is a single-channel processor. Place them in FDNConfig::loop_filter_configs, after the attenuation
 * filters and before the feedback matrix, using the multichannel banks built by the `MakeMultichannel…` factories
 * below. The feedback matrix then distributes the newly generated harmonics across every channel of the network,
 * which is what turns a per-channel waveshaper into a reverberation effect.
 *
 * All three are approximately energy preserving, which is what keeps the enclosing network stable, but none of them
 * is exactly so. Section 4 of the paper analyses this in detail; the individual classes document what it means for
 * them.
 * @{
 */

/** @brief A controllable full-wave rectifier, with optional antiderivative antialiasing.
 *
 * Implements equation (3) of the paper:
 *
 * \f[
 * y(n) = g_\mathrm{cfwr} \left( (1 - \alpha) x(n) + \alpha \left| x(n) \right| \right), \qquad
 * g_\mathrm{cfwr} = \sqrt{2 - 2 \left| \alpha - 1/2 \right|}.
 * \f]
 *
 * The parameter \f$\alpha\f$ blends between the untouched input and a full-wave rectifier, and \f$g_\mathrm{cfwr}\f$
 * compensates the power lost by folding the negative half of the waveform. It is 1 at \f$\alpha = 0\f$ and
 * \f$\alpha = 1\f$, where the operation is exactly energy preserving, and peaks at \f$\sqrt{2}\f$ at
 * \f$\alpha = 0.5\f$, where the rectifier is a half-wave rectifier.
 *
 * A rectifier generates every even harmonic of its input, so it aliases badly. Equation (4) of the paper replaces the
 * rectifier with its first-order antiderivative antialiasing approximation
 *
 * \f[
 * \left| x(n) \right| \approx \frac{1}{2}
 * \frac{x(n) \left| x(n) \right| - x(n-1) \left| x(n-1) \right|}{x(n) - x(n-1)},
 * \f]
 * which attenuates the aliased components that fall between the harmonics. The quotient is ill-conditioned when the
 * two consecutive samples are nearly equal, so it falls back to \f$\left| x(n) + x(n-1) \right| / 2\f$ there.
 *
 * @note The rectifier produces a dc component. Inside a feedback loop that component accumulates until the network
 * clips, so the processor follows itself with a dc blocker and a slow make-up gain by default. Only turn
 * ControllableFullWaveRectifierOptions::dc_block off when the processor is used as an insert effect.
 *
 * @note The peak gain of the operation is `ControllableFullWaveRectifierOptions::alpha`-dependent and reaches
 * \f$\sqrt{2}\f$ (+3 dB) at `alpha = 0.5`. The make-up gain of the dc blocker is capped, but it can still add up to
 * +12 dB on top of that on near-silent input. Budget for both in the attenuation filters of the enclosing network.
 *
 * @ingroup AudioProcessors
 */
class ControllableFullWaveRectifier : public AudioProcessor
{
  public:
    /** @brief Threshold below which the denominator of the antialiasing approximation is considered ill-conditioned.
     *
     * The reference implementation of the paper uses 1e-8 in double precision. sfFDN processes float32, where the
     * numerator `x(n)|x(n)| - x(n-1)|x(n-1)|` is a difference of two nearly equal quantities and has already lost
     * most of its significant digits by the time the denominator is that small. The threshold is therefore raised to
     * a value that float32 can actually resolve. See `.github/notes/shimmer-nonlinearities.md`.
     */
    static constexpr float kAntialiasingEpsilon = 1e-5f;

    /** @brief Constructs a controllable full-wave rectifier.
     * @param options The configuration options.
     * @throws std::invalid_argument if `alpha` is not in [0, 1], or if `dc_block` is true and `sample_rate` is not
     * strictly positive.
     */
    explicit ControllableFullWaveRectifier(const ControllableFullWaveRectifierOptions& options = {});

    ControllableFullWaveRectifier(const ControllableFullWaveRectifier& other);
    ControllableFullWaveRectifier& operator=(const ControllableFullWaveRectifier& other);
    ControllableFullWaveRectifier(ControllableFullWaveRectifier&&) noexcept;
    ControllableFullWaveRectifier& operator=(ControllableFullWaveRectifier&&) noexcept;
    ~ControllableFullWaveRectifier() override;

    /** @brief Sets the distortion amount.
     * @param alpha The distortion amount, in [0, 1].
     * @throws std::invalid_argument if `alpha` is not in [0, 1].
     * @note This also updates the compensation gain \f$g_\mathrm{cfwr}\f$.
     */
    void SetAlpha(float alpha);

    /** @brief Returns the distortion amount. */
    float GetAlpha() const noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the power compensation gain \f$g_\mathrm{cfwr}\f$ of the current distortion amount. */
    float GetCompensationGain() const noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a single sample. */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes the audio buffer.
     * @note The input and output buffers must have one channel and the same number of samples.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels this processor expects. This is always 1. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels this processor produces. This is always 1. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the antialiasing state and the dc blocker. The distortion amount is left untouched. */
    void Clear() override;

    /** @brief Creates a copy of the processor. The antialiasing and dc blocker state are carried over. */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float Rectify(float input) const noexcept SFFDN_NONBLOCKING;

    std::unique_ptr<DcBlocker> dc_blocker_;

    float alpha_;
    float compensation_gain_;
    bool antialiasing_;

    float prev_input_{0.f};
};

/** @brief A signal-dependent fractional delay filter.
 *
 * Implements the filter of Fig. 5 of the paper, after J. R. Pierce and S. A. Van Duyne, "A passive nonlinear digital
 * filter design which facilitates physics-based sound synthesis of highly nonlinear musical instruments", J. Acoust.
 * Soc. Amer., 1997, and V. Välimäki, T. Tolonen and M. Karjalainen, "Signal-dependent nonlinearities for physical
 * models using time-varying fractional delay filters", Proc. ICMC, 1998.
 *
 * The input is split into its positive and negative half-wave rectified components, each is delayed by a different
 * fractional amount, and the two are summed:
 *
 * \f[
 * p(n) = \max(x(n), 0), \qquad q(n) = \min(x(n), 0), \\
 * y(n) = d\,q(n) + (1 - d)\,q(n-1) + d\,p(n-2) + (1 - d)\,p(n-1).
 * \f]
 *
 * The two interpolators are linear and carry complementary weights, so the positive component is delayed by
 * \f$1 + d\f$ samples and the negative one by \f$1 - d\f$. The result keeps the fundamental and the overall amplitude
 * of a sinusoidal input but distorts it around its zero crossings, generating even harmonics. It is a milder effect
 * than ControllableFullWaveRectifier, and a good alternative to it on sustained input.
 *
 * @note The delayed positive and negative components overlap by up to one sample, so a zero-mean periodic signal can
 * lose up to one sample of energy per period. The operation is therefore slightly lossy rather than strictly energy
 * preserving, and the loss grows with frequency. It does not endanger the stability of the enclosing network. It
 * needs no dc blocker.
 *
 * @ingroup AudioProcessors
 */
class SignalDependentFractionalDelay : public AudioProcessor
{
  public:
    /** @brief Constructs a signal-dependent fractional delay.
     * @param options The configuration options.
     * @throws std::invalid_argument if `d` is not in [0, 1].
     */
    explicit SignalDependentFractionalDelay(const SignalDependentFractionalDelayOptions& options = {});

    /** @brief Sets the interpolation weight.
     * @param d The interpolation weight, in [0, 1].
     * @throws std::invalid_argument if `d` is not in [0, 1].
     */
    void SetD(float d);

    /** @brief Returns the interpolation weight. */
    float GetD() const noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a single sample. */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes the audio buffer.
     * @note The input and output buffers must have one channel and the same number of samples.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels this processor expects. This is always 1. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels this processor produces. This is always 1. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the delay state. The interpolation weight is left untouched. */
    void Clear() override;

    /** @brief Creates a copy of the processor. The delay state is carried over. */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    float d_;

    float prev_positive_{0.f};
    float prev_prev_positive_{0.f};
    float prev_negative_{0.f};
};

/** @brief A ring modulator.
 *
 * Implements equation (5) of the paper:
 *
 * \f[
 * y(n) = g_\mathrm{rm}\, x(n) \sin(2 \pi f_\mathrm{rm} n).
 * \f]
 *
 * The multiplication replaces the spectrum of the input with two sidebands, shifted by \f$\pm f_\mathrm{rm}\f$. Below
 * about 20 Hz the effect is heard as a tremolo, and above it as a change of timbre: harmonic when the modulation
 * frequency is an integer ratio of the input, and bell-like or metallic otherwise.
 *
 * The average power of a unit-amplitude sinusoid is one half, so the default amplitude of \f$\sqrt{2}\f$ makes the
 * operation energy preserving on average. That is an average over a modulation period, not a bound: the instantaneous
 * gain still reaches \f$\sqrt{2}\f$ at the peaks of the modulator. In a network with short, strongly recirculating
 * delay lines, or with a diagonally dominant feedback matrix, several channels can peak together for long enough to
 * make the energy grow. Lower the amplitude if that happens.
 *
 * @ingroup AudioProcessors
 */
class RingModulator : public AudioProcessor
{
  public:
    /** @brief Constructs a ring modulator.
     * @param options The configuration options.
     * @throws std::invalid_argument if the frequency is negative or not finite, if the amplitude is not finite, or if
     * the initial phase is not in [0, 1].
     */
    explicit RingModulator(const RingModulatorOptions& options = {});

    /** @brief Returns the modulation frequency, in cycles per sample. */
    float GetFrequency() const noexcept SFFDN_NONBLOCKING;

    /** @brief Returns the linear gain applied to the modulating sinusoid. */
    float GetAmplitude() const noexcept SFFDN_NONBLOCKING;

    /** @brief Processes a single sample. */
    float Tick(float input) noexcept SFFDN_NONBLOCKING;

    /** @brief Processes the audio buffer.
     * @note The input and output buffers must have one channel and the same number of samples.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of input channels this processor expects. This is always 1. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels this processor produces. This is always 1. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Restores the configured initial phase. */
    void Clear() override;

    /** @brief Creates a copy of the processor. The running modulation phase is carried over, not restarted. */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    SineWave lfo_;
};

/** @brief Builds a bank of controllable full-wave rectifiers, one per channel.
 * @param options The per-channel configurations. A `std::nullopt` entry leaves its channel unprocessed. The number of
 * channels of the returned processor is `options.channels.size()`.
 * @throws std::invalid_argument if any entry is invalid.
 */
std::unique_ptr<FilterBank> MakeMultichannelControllableFullWaveRectifier(
    const MultichannelControllableFullWaveRectifierOptions& options);

/** @brief Returns the options of a bank of controllable full-wave rectifiers covering every channel.
 * @param alpha The distortion amount applied to every active channel.
 * @param sample_rate The sample rate, in Hz.
 * @param channel_count The number of channels.
 * @param active_channel_count The number of channels that carry a rectifier, counted from the *last* channel. Zero
 * means every channel. The channels of an FDN are conventionally ordered by increasing delay length, and Section 6.1
 * of the paper recommends placing the nonlinearity behind the longer delay lines when a more gradually evolving
 * effect is wanted, so the active channels are taken from the end.
 */
MultichannelControllableFullWaveRectifierOptions MakeMultichannelControllableFullWaveRectifierOptions(
    float alpha, float sample_rate, uint32_t channel_count, uint32_t active_channel_count = 0);

/** @brief Builds a bank of signal-dependent fractional delays, one per channel.
 * @param options The per-channel configurations. A `std::nullopt` entry leaves its channel unprocessed.
 * @throws std::invalid_argument if any entry is invalid.
 */
std::unique_ptr<FilterBank> MakeMultichannelSignalDependentFractionalDelay(
    const MultichannelSignalDependentFractionalDelayOptions& options);

/** @brief Returns the options of a bank of signal-dependent fractional delays covering every channel.
 * @param d The interpolation weight applied to every active channel.
 * @param channel_count The number of channels.
 * @param active_channel_count The number of channels that carry a filter, counted from the last channel. Zero means
 * every channel.
 */
MultichannelSignalDependentFractionalDelayOptions MakeMultichannelSignalDependentFractionalDelayOptions(
    float d, uint32_t channel_count, uint32_t active_channel_count = 0);

/** @brief Builds a bank of ring modulators, one per channel.
 * @param options The per-channel configurations. A `std::nullopt` entry leaves its channel unprocessed.
 * @throws std::invalid_argument if any entry is invalid.
 */
std::unique_ptr<FilterBank> MakeMultichannelRingModulator(const MultichannelRingModulatorOptions& options);

/** @brief Returns the options of a decorrelated bank of ring modulators.
 * @param frequency The modulation frequency, in cycles per sample.
 * @param amplitude The linear gain applied to the modulating sinusoid.
 * @param channel_count The number of channels.
 * @param active_channel_count The number of channels that carry a modulator, counted from the last channel. Zero
 * means every channel.
 *
 * @note The initial phase of active channel `i` of `n` is `i / n`, so the channels do not reach the peak of their
 * modulator at the same instant. Phase-aligned modulators across every channel produce a much stronger tremolo,
 * because the feedback matrix then sums channels that are all being attenuated together.
 */
MultichannelRingModulatorOptions MakeMultichannelRingModulatorOptions(float frequency, float amplitude,
                                                                     uint32_t channel_count,
                                                                     uint32_t active_channel_count = 0);

/** @} */

} // namespace sfFDN
