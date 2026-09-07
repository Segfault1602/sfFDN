// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{
/** @defgroup MatrixGallery Matrix Gallery
 * @brief A collection of functions to generate various types of feedback matrices.
 * @{
 */

/** @brief Gets the scalar matrix type represented by a matrix generator recipe.
 * @param generator The matrix generator recipe.
 * @return The recipe's scalar matrix type.
 */
ScalarMatrixType GetMatrixType(const MatrixGeneratorOptions& generator) noexcept;

/** @brief Generates a square matrix of size mat_size x mat_size based on the specified generator recipe.
 * @param mat_size The size of the matrix (number of rows and columns).
 * @param generator The generator recipe.
 * @param seed Seed for every random gallery type.
 * @return A flat vector in row-major order: `matrix[row * mat_size + column]` is \f$A_{row,column}\f$.
 * The matrix maps input/source columns to output/destination rows (\f$y = A x\f$).
 *
 * ScalarMatrixType::VariableDiffusion uses diffusion 1. VariableDiffusionOptions supplies an explicit diffusion.
 *
 * Adapted from the implementation in the FDNTB toolbox by S. J. Schlecht:
 * https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/Generate/fdnMatrixGallery.m
 */
std::vector<float> GenerateMatrix(uint32_t mat_size, const MatrixGeneratorOptions& generator,
                                  uint32_t seed = kDefaultMatrixSeed);

/** @brief Generates a nested allpass matrix of size mat_size x mat_size as described in [1].
 * @param mat_size The size of the matrix (number of rows and columns). Must be a power of two.
 * @param seed Seed for random number generation.
 * @param input_gains [Optional|Out]: input gains for the allpass filters.
 * @param output_gains [Optional|Out]: output gains for the allpass filters.
 * @return A flat vector in row-major order: `matrix[row * mat_size + column]` is \f$A_{row,column}\f$.
 * The matrix maps input/source columns to output/destination rows (\f$y = A x\f$).
 *
 * @note [1] William G. Gardner; A real‐time multichannel room simulator. J. Acoust. Soc. Am. 1 October 1992; 92
 * (4_Supplement): 2395. https://doi.org/10.1121/1.404752
 */
std::vector<float> NestedAllpassMatrix(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed,
                                       std::span<float> input_gains = std::span<float>(),
                                       std::span<float> output_gains = std::span<float>());

/** @} */
} // namespace sfFDN