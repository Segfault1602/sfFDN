#pragma once

#include "sffdn/types.h"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace sfFDN
{
Eigen::MatrixXf RandN(uint32_t mat_size, uint32_t seed = 0);
Eigen::MatrixXf RandomOrthogonal(uint32_t mat_size, uint32_t seed = 0);
Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v);
Eigen::MatrixXf RandomHouseholder(uint32_t mat_size, uint32_t seed = 0);
Eigen::MatrixXf HadamardMatrix(uint32_t mat_size);
Eigen::MatrixXf CirculantMatrix(uint32_t mat_size, uint32_t seed = 0);
Eigen::MatrixXf AllpassMatrix(uint32_t mat_size, uint32_t seed = 0);

Eigen::MatrixXf GenerateMatrixInternal(uint32_t mat_size, sfFDN::ScalarMatrixType type, uint32_t seed,
                                       std::optional<float> arg = std::nullopt);

} // namespace sfFDN