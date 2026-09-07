#pragma once

#include "sffdn/types.h"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace sfFDN
{
Eigen::MatrixXf RandN(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed);
Eigen::MatrixXf RandomOrthogonal(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed);
Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v);
Eigen::MatrixXf RandomHouseholder(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed);
Eigen::MatrixXf HadamardMatrix(uint32_t mat_size);
Eigen::MatrixXf CirculantMatrix(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed);
Eigen::MatrixXf AllpassMatrix(uint32_t mat_size, uint32_t seed = kDefaultMatrixSeed);

Eigen::MatrixXf GenerateMatrixInternal(uint32_t mat_size, const MatrixGeneratorOptions& generator,
                                       uint32_t seed = kDefaultMatrixSeed);

} // namespace sfFDN