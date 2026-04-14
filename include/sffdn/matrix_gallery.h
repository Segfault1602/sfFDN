// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sfFDN
{
/** @defgroup MatrixGallery Matrix Gallery
 * @brief A collection of functions to generate various types of feedback matrices.
 * @{
 */

/** @brief Generates a square matrix of size mat_size x mat_size based on the specified type.
 * @param mat_size The size of the matrix (number of rows and columns).
 * @param type The type of matrix to generate.
 * @param seed Seed for random number generation (used for Random and RandomHouseholder types).
 * @param arg Optional argument for certain matrix types.
 * @return A flat vector containing the matrix elements in col-major order.
 *
 * @note For the VariableDiffusion type, the optional argument 'arg' specifies the sparsity level (0 <= arg <= 1).
 * @note Adapted from the implementation in the FDNTB toolbox by S. J. Schlecht:
 * https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/Generate/fdnMatrixGallery.m
 */
std::vector<float> GenerateMatrix(uint32_t mat_size, ScalarMatrixType type, uint32_t seed = 0,
                                  std::optional<float> arg = std::nullopt);

/** @brief Generates a nested allpass matrix of size mat_size x mat_size as described in [1].
 * @param mat_size The size of the matrix (number of rows and columns). Must be a power of two.
 * @param seed Seed for random number generation.
 * @param input_gains [Optional|Out]: input gains for the allpass filters.
 * @param output_gains [Optional|Out]: output gains for the allpass filters.
 * @return A flat vector containing the matrix elements in col-major order.
 *
 * @note [1] William G. Gardner; A real‐time multichannel room simulator. J. Acoust. Soc. Am. 1 October 1992; 92
 * (4_Supplement): 2395. https://doi.org/10.1121/1.404752
 */
std::vector<float> NestedAllpassMatrix(uint32_t mat_size, uint32_t seed = 0,
                                       std::span<float> input_gains = std::span<float>(),
                                       std::span<float> output_gains = std::span<float>());

/**
 * @brief Constructs a Cascaded feedback matrix.
 *
 * @param channel_count Number of channels
 * @param stage_count Number of stages
 * @param sparsity Sparsity level (>= 1)
 * @param gain_per_samples Gain per sample (default: 1.0)
 * @return CascadedFeedbackMatrixInfo
 */
CascadedFeedbackMatrixOptions ConstructCascadedFeedbackMatrix(uint32_t channel_count, uint32_t stage_count,
                                                              float sparsity, ScalarMatrixType type,
                                                              float gain_per_samples = 1.f);

/** @} */
} // namespace sfFDN