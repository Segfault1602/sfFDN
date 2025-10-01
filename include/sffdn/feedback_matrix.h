// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "audio_processor.h"
#include "matrix_gallery.h"

namespace sfFDN
{
/** @brief A scalar feedback matrix processor.
 * This processor applies a square feedback matrix to the input audio buffer.
 * The matrix is defined as a flat array in column-major order.
 *
 * @ingroup AudioProcessors
 */
class ScalarFeedbackMatrix : public AudioProcessor
{
  public:
    /** @brief Constructs a scalar feedback matrix.
     * @param order The size of the square matrix, where the size == number of rows == number of columns.
     * @param type The type of matrix to create.
     */
    ScalarFeedbackMatrix(uint32_t order = 4, ScalarMatrixType type = ScalarMatrixType::Identity);

    /** @brief Constructs a scalar feedback matrix from a custom matrix.
     * @param order The size of the square matrix, where the size == number of rows == number of columns.
     * @param matrix A span representing the matrix coefficients in column-major order. The span must be of size
     * `order * order`.
     */
    ScalarFeedbackMatrix(uint32_t order, std::span<const float> matrix);

    ~ScalarFeedbackMatrix() override;
    ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other);
    ScalarFeedbackMatrix& operator=(const ScalarFeedbackMatrix& other);
    ScalarFeedbackMatrix(ScalarFeedbackMatrix&& other) noexcept;
    ScalarFeedbackMatrix& operator=(ScalarFeedbackMatrix&& other) noexcept;

    /** @brief Sets the matrix coefficients.
     * @param matrix A span representing the matrix coefficients in column-major order. The span must be of size
     * `order * order`.
     * @return true if the matrix was set successfully, false otherwise (e.g. if the size is incorrect).
     */
    bool SetMatrix(const std::span<const float> matrix);

    /**
     * @brief Get the Matrix object
     *
     * @param matrix
     * @return false if the span is not the correct size.
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
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /** @brief Returns the size of the square matrix (number of rows/columns).
     * @return The size of the matrix.
     */
    uint32_t GetSize() const;

    /**
     * @brief Get a specific coefficient from the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @return the coefficient at the specified row and column
     */
    float GetCoefficient(uint32_t row, uint32_t col) const;

    /** @brief Returns the number of input channels supported by the processor. */
    uint32_t InputChannelCount() const override;

    /** @brief Returns the number of output channels produced by the processor. */
    uint32_t OutputChannelCount() const override;

    /** @brief Clears the internal state of the processor.
     * This is a no-op for ScalarFeedbackMatrix as it has no internal state.
     */
    void Clear() override;

    /** @brief Creates a copy of the scalar feedback matrix.
     * @return A unique pointer to the cloned scalar feedback matrix.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    class ScalarFeedbackMatrixImpl;
    std::unique_ptr<ScalarFeedbackMatrixImpl> impl_;
};

} // namespace sfFDN