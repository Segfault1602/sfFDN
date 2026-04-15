// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "sffdn/audio_processor.h"

#include <cstdint>
#include <span>

namespace sfFDN
{
// Forward declaration
class ScalarFeedbackMatrix;

/** @brief DelayMatrix implementation as presented in [1].
 *
 * [1] S. J. Schlecht and E. A. P. Habets, "Dense Reverberation with Delay Feedback Matrices," in 2019 IEEE Workshop on
 * Applications of Signal Processing to Audio and Acoustics (WASPAA), Oct. 2019, pp. 150–154.
 * doi: 10.1109/WASPAA.2019.8937284.
 *
 * @ingroup AudioProcessors
 */
class DelayMatrix : public AudioProcessor
{
  public:
    /** @brief Constructs a DelayMatrix with the specified size and delay values.
     * @param order the size of the square matrix (order x order)
     * @param delays the delay values for each channel. The size of the delays span must match order.
     */
    DelayMatrix(uint32_t order, std::span<const uint32_t> delays, const ScalarFeedbackMatrix& mixing_matrix);

    ~DelayMatrix() override;

    DelayMatrix(const DelayMatrix&);
    DelayMatrix& operator=(const DelayMatrix&);

    DelayMatrix(DelayMatrix&&) noexcept;
    DelayMatrix& operator=(DelayMatrix&&) noexcept;

    /** @brief Processes the input audio buffer through the delay matrix.
     * @param input the input audio buffer
     * @param output the output audio buffer
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the number of input channels this processor expects. */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels this processor produces. */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal delay buffers.
     * This sets all delay buffers to zero without changing the delay values or mixing matrix.
     */
    void Clear() override;

    /** @brief Prints information about the delay matrix. For debugging purposes. */
    void PrintInfo() const;

    /** @brief Creates a copy of the delay matrix.
     * @return A unique pointer to the cloned delay matrix.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class DelayMatrixImpl;
    std::unique_ptr<DelayMatrixImpl> impl_;

    DelayMatrix() = default; // Default constructor used internally by Clone()
};
} // namespace sfFDN