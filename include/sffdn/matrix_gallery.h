// Copyright (C) 2025 Alexandre St-Onge
// SPDX-License-Identifier: MIT
#pragma once

#include <span>
#include <vector>

namespace sfFDN
{
enum class ScalarMatrixType : uint8_t
{
    Identity = 0,
    Random = 1,
    Householder = 2,
    RandomHouseholder = 3,
    Hadamard = 4,
    Circulant = 5,
    Allpass = 6,
    NestedAllpass = 7,
    Count
};

// Generates a square matrix of size N x N based on the specified type.
std::vector<float> GenerateMatrix(size_t N, ScalarMatrixType type, uint32_t seed = 0);

std::vector<float> NestedAllpassMatrix(size_t N, uint32_t seed = 0, std::span<float> input_gains = std::span<float>(),
                                       std::span<float> output_gains = std::span<float>());

struct CascadedFeedbackMatrixInfo
{
    size_t N;                     // Number of channels
    size_t K;                     // Number of stages
    std::vector<uint32_t> delays; // Delays, size: (K - 1) x N
    std::vector<float> matrices;  // Feedback matrices, size: K x N x N
};

/**
 * @brief Constructs a Cascaded feedback matrix.
 *
 * @param N Number of channels
 * @param K Number of stages
 * @param sparsity Sparsity level (>= 1)
 * @param gain_per_samples Gain per sample (default: 1.0)
 * @return CascadedFeedbackMatrixInfo
 */
CascadedFeedbackMatrixInfo ConstructCascadedFeedbackMatrix(size_t N, size_t K, float sparsity, ScalarMatrixType type,
                                                           float gain_per_samples = 1.f);

} // namespace sfFDN