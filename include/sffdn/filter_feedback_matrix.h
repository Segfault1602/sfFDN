// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "delaybank.h"
#include "feedback_matrix.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{

/**
 * @brief A filter feedback matrix processor.
 * This processor implements a filter feedback matrix as described in [1]
 *
 * Structure: Input──[D₀]──[U₁]──[D₂]──[U₂]──...──[Uₖ]──[Dₖ]──Output
 * Where: Dᵢ = delay bank, Uᵢ = mixing matrix, K = number of stages
 *
 * [1] S. J. Schlecht and E. A. P. Habets, “Scattering in feedback delay networks,” IEEE/ACM Transactions on Audio,
 * Speech, and Language Processing, vol. 28, June 2020.
 *
 * @ingroup AudioProcessors
 */
class FilterFeedbackMatrix : public AudioProcessor
{
  public:
    /** @brief Constructs a filter feedback matrix with a specified number of channels.
     * @param options The information structure containing channel and stage counts, delays, and matrices.
     */
    FilterFeedbackMatrix(const CascadedFeedbackMatrixOptions& options);

    ~FilterFeedbackMatrix() override = default;

    FilterFeedbackMatrix(const FilterFeedbackMatrix& other) = delete;
    FilterFeedbackMatrix& operator=(const FilterFeedbackMatrix& other) = delete;

    /** @brief Move constructor for the filter feedback matrix.
     * @param other The filter feedback matrix to move from.
     */
    FilterFeedbackMatrix(FilterFeedbackMatrix&& other) noexcept;

    /** @brief Move assignment operator for the filter feedback matrix.
     * @param other The filter feedback matrix to move from.
     * @return A reference to the assigned filter feedback matrix.
     */
    FilterFeedbackMatrix& operator=(FilterFeedbackMatrix&& other) noexcept;

    /**
     * @brief Processes the input audio buffer and produces the output audio buffer.
     *
     * @param input AudioBuffer containing the input audio data. The number of channels must match the channel count.
     * @param output AudioBuffer containing the output audio data. The number of channels must match the channel count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING override;

    /**
     * @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return channel_count_;
    }

    /**
     * @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const noexcept SFFDN_NONBLOCKING override
    {
        return channel_count_;
    }

    /** @brief Clears the internal state of the processor.
     * This function clears the internal state of all delay banks.
     */
    void Clear() override;

    /** @brief Prints information about the filter feedback matrix to the standard output. */
    void PrintInfo() const;

    /**
     * @brief Retrieves the coefficients of the first feedback matrix in the cascade.
     *
     * @param matrix A span of size `channel_count_ * channel_count_` to fill with the coefficients of the first
     *   feedback matrix in row-major order: filled[row*N+col] = A[row,col].
     * @return true if the coefficients were retrieved successfully, false otherwise (e.g. if the size is incorrect).
     */
    bool GetFirstMatrix(std::span<float> matrix) const;

    /** @brief Creates a copy of the filter feedback matrix.
     * @return A unique pointer to the cloned filter feedback matrix.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t channel_count_;
    std::vector<DelayBank> delaybanks_;
    std::vector<ScalarFeedbackMatrix> matrix_;

    FilterFeedbackMatrix();
};

} // namespace sfFDN