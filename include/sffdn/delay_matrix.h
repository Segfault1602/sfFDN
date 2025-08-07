// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <span>

#include "sffdn/feedback_matrix.h"

namespace sfFDN
{
/// @brief DelayMatrix implementation as presented in [1].
///
/// [1] S. J. Schlecht and E. A. P. Habets, “Dense Reverberation with Delay Feedback Matrices,” in 2019 IEEE Workshop on
/// Applications of Signal Processing to Audio and Acoustics (WASPAA), Oct. 2019, pp. 150–154.
/// doi: 10.1109/WASPAA.2019.8937284.
class DelayMatrix : public AudioProcessor
{
  public:
    /// @brief Constructs a DelayMatrix with the specified size and delay values.
    /// @param N the size of the matrix
    /// @param delays the delay values for each channel. The size of the delays span must match N.
    DelayMatrix(uint32_t N, std::span<const uint32_t> delays, const ScalarFeedbackMatrix& mixing_matrix);

    ~DelayMatrix() override;

    DelayMatrix(const DelayMatrix&) = delete;
    DelayMatrix& operator=(const DelayMatrix&) = delete;
    DelayMatrix(DelayMatrix&&) = default;
    DelayMatrix& operator=(DelayMatrix&&) = default;

    /// @brief Clears the internal delay buffers.
    /// This sets all delay buffers to zero.
    void Clear();

    /// @brief Processes the input audio buffer through the delay matrix.
    /// @param input the input audio buffer
    /// @param output the output audio buffer
    void Process(const AudioBuffer& input, AudioBuffer& output) override;

    uint32_t InputChannelCount() const override;

    uint32_t OutputChannelCount() const override;

    /// @brief Prints information about the delay matrix. For debugging purposes.
    void PrintInfo() const;

  private:
    class DelayMatrixImpl;
    std::unique_ptr<DelayMatrixImpl> impl_;
};
} // namespace sfFDN