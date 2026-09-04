// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "audio_processor.h"
#include "matrix_gallery.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sfFDN
{
/** @brief A scalar feedback matrix processor.
 * This processor applies a square feedback matrix to the input audio buffer.
 * The public matrix convention is **row-major**: flat[row*N+col] = A[row,col], y = A*x.
 *
 * @ingroup AudioProcessors
 */
class ScalarFeedbackMatrix : public AudioProcessor
{
  public:
    /** @brief Constructs a scalar feedback matrix.
     * @param config The configuration for the scalar feedback matrix.
     */
    ScalarFeedbackMatrix(const ScalarFeedbackMatrixOptions& config);

    /** @brief Sets the matrix coefficients.
     * @param matrix A span of matrix coefficients in row-major order: flat[row*N+col] = A[row,col].
     *   The span must contain exactly `order * order` elements (where `order` is the value returned by
     *   GetSize()). The size of the matrix cannot be changed by this method; passing the wrong number of
     *   elements is rejected and leaves the current state unchanged.
     * @return true if the matrix was set successfully, false otherwise (e.g. if the size is incorrect).
     */
    bool SetMatrix(std::span<const float> matrix);

    /**
     * @brief Retrieves the matrix coefficients.
     *
     * @param matrix A span of size `order * order` to fill with the matrix coefficients in row-major order:
     *   filled[row*N+col] = A[row,col].
     * @return false if the span is not exactly `order * order` in size.
     */
    bool GetMatrix(std::span<float> matrix) const;

    /**
     * @brief Processes the input audio buffer through the feedback matrix.
     *
     * @param input AudioBuffer containing the input audio data. The number of channels must match the matrix order
     * returned by GetSize().
     * @param output AudioBuffer to fill with the processed audio data. The number of channels must match the matrix
     * order.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the size of the square matrix (number of rows/columns).
     * @return The size of the matrix.
     */
    uint32_t GetSize() const;

    /**
     * @brief Get a specific coefficient from the matrix.
     * Uses the row-major convention: A[row,col] = flat[row*N+col].
     *
     * @param row The row index (destination)
     * @param col The column index (source)
     * @return the coefficient at the specified row and column
     */
    float GetCoefficient(uint32_t row, uint32_t col) const;

    /** @brief Returns the number of input channels supported by the processor. */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Returns the number of output channels produced by the processor. */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override;

    /** @brief Clears the internal state of the processor.
     * This is a no-op for ScalarFeedbackMatrix as it has no internal state.
     */
    void Clear() override;

    /** @brief Creates a copy of the scalar feedback matrix.
     * @return A unique pointer to the cloned scalar feedback matrix.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t order_;
    ScalarMatrixType matrix_type_;
    std::vector<float> matrix_data_;
};

} // namespace sfFDN