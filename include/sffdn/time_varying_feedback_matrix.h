// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_processor.h"
#include "sffdn/oscillator.h"
#include "sffdn/types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sfFDN
{

namespace detail
{
class TimeVaryingFeedbackMatrixTestAccess;
}

/** @brief An orthogonal time-varying feedback matrix for an FDN.
 *
 * At sample `n`, the construction is:
 * - Hadamard mode: \f$A(n) = H^T \mathop{\mathrm{blockdiag}}(R(\theta_k(n))) H\f$.
 * - RealSchur mode: \f$A(n) = V \mathop{\mathrm{blockdiag}}(R(\theta_k(n))) V^T\f$.
 *
 * Each factor is orthogonal, so \f$A(n)\f$ is exactly orthogonal at every sample. This makes a time-varying FDN
 * stable and energy-preserving.
 *
 * @par Modulation units and recommended setup
 * `ModulationOptions::amplitude` is normalized, with `|amplitude| <= 1`, following the AES paper's convention.
 * It is multiplied by π exactly once internally to obtain the peak angular deviation in radians. Thus,
 * `amplitude = 0.7` means a peak deviation of `0.7π`, approximately 2.2 radians, not 0.7 radians. The JASA paper
 * instead uses the angular convention `μ_A <= π`. LFO frequency is in cycles per sample: 1 Hz at 48 kHz is
 * `1.0f / 48000.0f`.
 *
 * Schlecht and Habets (2015, JASA 138(3) and AES Convention paper 9255) recommend an approximately 1 Hz modulation
 * frequency, an amplitude of approximately 0.7, random initial phases, and per-block frequencies spread by about
 * ±50%. Frequencies below 0.2 Hz make modulation imperceptible; above 2 Hz they introduce obvious periodic temporal
 * patterns and detuning, although up to 4 Hz can still sound smooth and natural. Do not modulate every block
 * synchronously: the papers report easily perceivable beating.
 *
 * @par Limitations
 * Hadamard mode requires a power-of-two `matrix_size`; RealSchur mode accepts any even size. Both reject odd sizes
 * because an odd-dimensional orthogonal matrix necessarily has a static real eigenvalue. Modulation is full-band.
 *
 * @ingroup AudioProcessors
 */
class TimeVaryingFeedbackMatrix : public AudioProcessor
{
  public:
    /** @brief Constructs a time-varying feedback matrix.
     * @param options The configuration options for the feedback matrix.
     * @throws std::invalid_argument If the matrix size or mode is invalid, the Hadamard size is not a power of two,
     * the modulation count is invalid, a modulation parameter is non-finite, an initial phase is outside [0, 1],
     * or an amplitude is outside [-1, 1].
     */
    explicit TimeVaryingFeedbackMatrix(const TimeVaryingFeedbackMatrixOptions& options);

    /** @brief Sets the modulation options for each rotation block.
     * @param modulation_configs The modulation options, one per rotation block.
     * An empty span disables modulation. Otherwise, the span must contain exactly one configuration per rotation block.
     */
    void SetModulation(std::span<const ModulationOptions> modulation_configs);

    /** @brief Sets the LFO frequency for each rotation block.
     * @param frequencies The normalized LFO frequencies, one per rotation block.
     * Frequencies are finite values in cycles per sample; for example, 1 Hz at 48 kHz is `1.0f / 48000.0f`.
     */
    void SetLfoFrequency(std::span<const float> frequencies);

    /** @brief Sets the LFO amplitude for each rotation block.
     * @param amplitudes The normalized LFO amplitudes, one per rotation block.
     * Each amplitude must be in [-1, 1] and is multiplied by π once internally to obtain its peak deviation in
     * radians. For example, 0.7 denotes `0.7π`, not 0.7 radians.
     */
    void SetLfoAmplitude(std::span<const float> amplitudes);

    /** @brief Sets the LFO phase offset for each rotation block.
     * @param phase_offsets The normalized LFO phase offsets, one per rotation block.
     * Each offset is finite and in [0, 1] cycles, where 1 is a full cycle.
     */
    void SetLfoPhaseOffset(std::span<const float> phase_offsets);

    /** @brief Sets the base rotation angle for each rotation block.
     * @param radians The finite base angles in radians, one per rotation block. Angles are range-reduced to [-π, π].
     */
    void SetBaseAngles(std::span<const float> radians);

    /** @brief Processes the audio buffer.
     * @param input The input audio buffer.
     * @param output The output audio buffer.
     * The input and output channel counts must both equal the matrix size, and their sample counts must match.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the number of input channels.
     * @return The matrix size.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the number of output channels.
     * @return The matrix size.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Gets the number of 2x2 rotation blocks.
     * @return The number of independently modulatable rotation blocks.
     * @note Query this after construction before sizing a RealSchur `time_varying_config`, because its block count
     * depends on the constructed basis. Hadamard mode always has `matrix_size / 2` blocks.
     */
    uint32_t RotationBlockCount() const noexcept SFFDN_NONBLOCKING;

    /** @brief Materializes the feedback matrix that Process applies at a given sample index.
     * @param matrix Destination for `matrix_size * matrix_size` values in column-major order, matching
     * ScalarFeedbackMatrix::GetMatrix.
     * @param sample_index The index, counted from the start of an unprocessed stream, of the sample whose matrix is
     * wanted. Zero yields the matrix applied to the very first sample.
     * @return False if the span does not contain exactly `matrix_size * matrix_size` values.
     *
     * This is a pure query: it neither reads nor mutates the running LFO phases, so its result depends only on the
     * configuration and `sample_index`. It allocates and is not realtime-safe; call it from a control or UI thread.
     * Because it ignores the processing state, it is safe to call concurrently with Process.
     */
    bool GetMatrix(std::span<float> matrix, uint64_t sample_index = 0) const;

    /** @brief Resets the phase of all LFOs to their configured phase offsets. */
    void Clear() override;

    /** @brief Creates a copy of the time-varying feedback matrix.
     * @return A unique pointer to the cloned processor.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    friend class detail::TimeVaryingFeedbackMatrixTestAccess;

    TimeVaryingFeedbackMatrix(const TimeVaryingFeedbackMatrixOptions& options,
                              std::span<const float> custom_base_matrix);

    uint32_t order_;
    TimeVaryingMatrixMode mode_;
    std::vector<float> base_angles_;
    std::vector<SineWave> lfos_;
    std::vector<float> lfo_phases_;
    std::vector<float> schur_basis_;
    std::vector<uint32_t> rotation_starts_;
    std::vector<float> scalar_signs_;
    std::vector<float> scratch_;
};

} // namespace sfFDN
