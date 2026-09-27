// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "kronecker_feedback_matrix.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sfFDN
{

/** @brief Sine-modulated Kronecker feedback matrix.
 *
 * Owns a static KroneckerFeedbackMatrix and supplies per-sample angular offsets through its real-time processing hook.
 * Base angles are radians. Modulation frequency is cycles per sample, amplitude is a fraction of pi, and initial phase
 * is cycles. A fresh or cleared processor evaluates initial phase at sample zero before advancing.
 *
 * @ingroup AudioProcessors
 */
class TimeVaryingKroneckerFeedbackMatrix : public AudioProcessor
{
  public:
    explicit TimeVaryingKroneckerFeedbackMatrix(const TimeVaryingKroneckerFeedbackMatrixOptions& options);

    TimeVaryingKroneckerFeedbackMatrix(const TimeVaryingKroneckerFeedbackMatrix&) = delete;
    TimeVaryingKroneckerFeedbackMatrix& operator=(const TimeVaryingKroneckerFeedbackMatrix&) = delete;
    TimeVaryingKroneckerFeedbackMatrix(TimeVaryingKroneckerFeedbackMatrix&&) noexcept = default;
    TimeVaryingKroneckerFeedbackMatrix& operator=(TimeVaryingKroneckerFeedbackMatrix&&) noexcept = default;
    ~TimeVaryingKroneckerFeedbackMatrix() override = default;

    /** @brief Sets the owned static matrix's base angles.
     * @throws std::invalid_argument Unless the span contains exactly M angles.
     *
     * This is a setup/control operation and must not run concurrently with Process().
     */
    void SetAngles(std::span<const float> radians);

    /** @brief Replaces the per-stage sine modulation configuration.
     * @throws std::invalid_argument Unless the span is empty or contains exactly M valid entries.
     *
     * An empty span disables modulation. Existing phase positions are retained, including while modulation is disabled.
     * This is a setup/control operation and must not run concurrently with Process().
     */
    void SetTimeVaryingConfig(std::span<const ModulationOptions> modulation_configs);

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t StageCount() const noexcept SFFDN_NONBLOCKING;

    /** @brief Materializes the reset-relative matrix at `sample_index` in row-major order. */
    bool GetMatrix(std::span<float> matrix, uint64_t sample_index = 0) const;

    void Clear() override;
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    static constexpr size_t kAngleChunkSize = 128U;

    TimeVaryingKroneckerFeedbackMatrixOptions options_;
    KroneckerFeedbackMatrix matrix_;
    std::vector<ModulationOptions> modulation_configs_;
    std::vector<double> phase_increments_;
    std::vector<double> phases_;
    std::vector<float> angle_offsets_;
};

} // namespace sfFDN
