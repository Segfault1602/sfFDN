// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

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

/** @brief Represents the type of a scalar matrix.
 *
 * [1] D. Rocchesso and J. O. Smith, “Circulant and elliptic feedback delay networks for artificial reverberation,” IEEE
 Transactions on Speech and Audio Processing, vol. 5, no. 1, pp. 51–63, Jan. 1997, doi: 10.1109/89.554269.\n
* [2] S. J. Schlecht, “FDNTB: the feedback delay network toolbox,” 23rd International Conference on Digital Audio
Effects (DAFx2020), 2020.\n
* [3] O. Das, E. K. Canfield-Dafilou, and J. S. Abel, “On The Behavior of Delay Network Reverberator Modes,” in 2019
IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), Oct. 2019, pp. 50–54.
doi: 10.1109/WASPAA.2019.8937260.
*/
enum class ScalarMatrixType : uint8_t
{
    Identity = 0,          /**< Identity matrix. */
    Random = 1,            /**< Random orthogonal matrix. */
    Householder = 2,       /**< Householder matrix. */
    RandomHouseholder = 3, /**< Random Householder matrix. */
    Hadamard = 4,          /**< Hadamard matrix. */
    Circulant = 5,         /**< Circulant matrix as described in [1] */
    Allpass = 6,           /**< Allpass matrix. See [2]*/
    NestedAllpass = 7,     /**< Nested Allpass matrix. See [2] */
    VariableDiffusion = 8, /**< Variable diffusion matrix as described in [3] */
    Count = 9
};

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
 * @brief Constructs a Cascaded feedback matrix.
 *
 * @param channel_count Number of channels
 * @param stage_count Number of stages
 * @param sparsity Sparsity level (>= 1)
 * @param gain_per_samples Gain per sample (default: 1.0)
 * @return CascadedFeedbackMatrixInfo
 */
CascadedFeedbackMatrixInfo ConstructCascadedFeedbackMatrix(uint32_t channel_count, uint32_t stage_count, float sparsity,
                                                           ScalarMatrixType type, float gain_per_samples = 1.f);

/** @} */
} // namespace sfFDN