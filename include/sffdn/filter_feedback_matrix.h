// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "delaybank.h"
#include "feedback_matrix.h"
#include "matrix_gallery.h"

namespace sfFDN
{

/** @brief Information structure for constructing a cascaded feedback matrix (also known as a filter feedback matrix).
 */
struct CascadedFeedbackMatrixInfo
{
    uint32_t channel_count;       /**< Number of channels */
    uint32_t stage_count;         /**< Number of stages */
    std::vector<uint32_t> delays; /**< Delays, size: stage_count x N */
    std::vector<float> matrices;  /**< Feedback matrices, size: K x N x N */
};

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
     * @param info The information structure containing channel and stage counts, delays, and matrices.
     */
    FilterFeedbackMatrix(const CascadedFeedbackMatrixInfo& info);

    ~FilterFeedbackMatrix() override = default;
    FilterFeedbackMatrix(const FilterFeedbackMatrix& other);
    FilterFeedbackMatrix& operator=(const FilterFeedbackMatrix& other);
    FilterFeedbackMatrix(FilterFeedbackMatrix&& other) noexcept;
    FilterFeedbackMatrix& operator=(FilterFeedbackMatrix&& other) noexcept;

    /**
     * @brief Processes the input audio buffer and produces the output audio buffer.
     *
     * @param input AudioBuffer containing the input audio data. The number of channels must match the channel count.
     * @param output AudioBuffer containing the output audio data. The number of channels must match the channel count.
     */
    void Process(const AudioBuffer& input, AudioBuffer& output) noexcept override;

    /**
     * @brief Returns the number of input channels supported by this processor.
     *
     * @return The number of input channels.
     */
    uint32_t InputChannelCount() const override
    {
        return channel_count_;
    }

    /**
     * @brief Returns the number of output channels produced by this processor.
     *
     * @return The number of output channels.
     */
    uint32_t OutputChannelCount() const override
    {
        return channel_count_;
    }

    /** @brief Clears the internal state of the processor.
     * This function clears the internal state of all delay banks.
     */
    void Clear() override;

    /** @brief Prints information about the filter feedback matrix to the standard output. */
    void PrintInfo() const;

    // TODO: this is just for the GUI in FDNSandbox
    bool GetFirstMatrix(std::span<float> matrix) const;

    /** @brief Creates a copy of the filter feedback matrix.
     * @return A unique pointer to the cloned filter feedback matrix.
     */
    std::unique_ptr<AudioProcessor> Clone() const override;

  private:
    uint32_t channel_count_;
    std::vector<DelayBank> delaybanks_;
    std::vector<ScalarFeedbackMatrix> matrix_;
};

} // namespace sfFDN