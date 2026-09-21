// Copyright (C) 2026 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sfFDN
{

/** @brief Fast static orthogonal Kronecker feedback matrix.
 *
 * For a matrix of order `N = 2^M`, stage zero acts on adjacent channels, stage one on pairs separated by two
 * channels, and so on. Rotation and reflection kernels may be selected independently for each stage.
 *
 * Process supports disjoint input/output views and exact in-place operation. Arbitrary partial overlap is unsupported.
 * It overwrites the output and performs no allocation.
 *
 * @ingroup AudioProcessors
 */
class KroneckerFeedbackMatrix : public AudioProcessor
{
  public:
    explicit KroneckerFeedbackMatrix(const KroneckerFeedbackMatrixOptions& options);

    KroneckerFeedbackMatrix(const KroneckerFeedbackMatrix&) = delete;
    KroneckerFeedbackMatrix& operator=(const KroneckerFeedbackMatrix&) = delete;
    KroneckerFeedbackMatrix(KroneckerFeedbackMatrix&&) noexcept = default;
    KroneckerFeedbackMatrix& operator=(KroneckerFeedbackMatrix&&) noexcept = default;
    ~KroneckerFeedbackMatrix() override = default;

    /** @brief Sets all M stage angles.
     * @param radians One angle in radians per stage.
     * @throws std::invalid_argument Unless the span contains exactly M angles.
     *
     * Angles are periodically wrapped to [-pi, pi].
     */
    void SetAngles(std::span<const float> radians);

    /** @brief Processes a block using per-stage angular offsets from the configured angles.
     * @param input Input audio.
     * @param output Output audio.
     * @param angle_offsets Stage-major offsets with `StageCount() * input.SampleCount()` values.
     * @param varying_stage_mask Bit `stage` is set when every sample in that stage row is used. A clear bit means the
     * first value in the row is constant for the whole block.
     *
     * The buffer dimensions, exact offset count, and mask bounds are preconditions. Empty blocks return without
     * reading `angle_offsets`. The resulting configured angle plus offset must stay inside [-2pi, 2pi].
     */
    void ProcessWithAngleOffsets(const AudioBuffer& input, AudioBuffer& output, std::span<const float> angle_offsets,
                                 uint32_t varying_stage_mask) const noexcept SFFDN_NONBLOCKING;

    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;
    uint32_t StageCount() const noexcept SFFDN_NONBLOCKING;

    /** @brief Materializes the configured fixed matrix in row-major order.
     * @return False without modifying `matrix` unless it contains exactly N squared elements.
     */
    bool GetMatrix(std::span<float> matrix) const;

    /** @brief No-op because the static matrix has no processing state. */
    void Clear() override;
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t order_;
    uint32_t stage_count_;
    std::vector<float> angles_;
    std::vector<KroneckerKernelType> kernel_types_;
    std::vector<float> sines_;
    std::vector<float> cosines_;
};

} // namespace sfFDN
